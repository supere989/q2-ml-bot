#!/usr/bin/env python3
"""reward_graph.py — full reward enumeration graph.

Two panels:
  (A) every configured reward term to scale, grouped by bucket, sign-coloured
      — the DESIGNED reward surface (runtime weights from the live env).
  (B) measured per-episode contribution by bucket — the ACTUAL surface, which
      exposes scale skew (one mis-scaled term can dwarf all shaping).

Run with the SAME env the training run uses so weights match. Empirical
numbers are passed in (pulled from TensorBoard) or default to the 2026-06-12
run-B snapshot.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── (A) configured weights, grouped by bucket ───────────────────────────────
# (label, attr_or_base_key, sign)  sign = intended direction (+reward/−penalty)
import harness.protocol as P
import harness.spatial as S
cfg = S.VoxelSpatialReward.from_env()

def w(attr):  # spatial config weight
    return float(getattr(cfg, attr))

BUCKETS = {
    "BASE game (per event)": [
        ("damage_dealt", P.R_DAMAGE_DEALT, +1), ("kill", P.R_KILL, +1),
        ("damage_taken", P.R_DAMAGE_TAKEN, -1), ("death", P.R_DEATH, -1),
        ("item", P.R_ITEM, +1), ("hook", P.R_HOOK, +1),
    ],
    "EXPLORE (per step)": [
        ("new_cell", w("new_cell_reward"), +1),
        ("stagnation", w("stagnation_penalty"), -1),
    ],
    "MEMORY / voxel (per step)": [
        ("engagement", w("session_memory_engagement_reward"), +1),
        ("opportunity", w("session_memory_opportunity_reward"), +1),
        ("threat", w("session_memory_threat_penalty"), -1),
        ("death_aversion", w("session_memory_death_aversion"), -1),
        ("self_fire", w("session_memory_self_fire_penalty"), -1),
        ("camp", w("session_memory_camp_penalty"), -1),
    ],
    "THREAT / combat (per step)": [
        ("engagement", w("threat_engagement_reward"), +1),
        ("aim", w("threat_aim_reward"), +1), ("fire", w("threat_fire_reward"), +1),
        ("damage", w("threat_damage_reward"), +1), ("kill", w("threat_kill_reward"), +1),
        ("ignore", w("threat_ignore_penalty"), -1), ("unready", w("threat_unready_penalty"), -1),
    ],
    "FIRE / HOOK discipline (per step)": [
        ("aim_align", w("aim_alignment_reward"), +1), ("engage_prox", w("engagement_reward"), +1),
        ("fire_cost", w("fire_cost"), -1), ("fire_unseen", w("fire_unseen_penalty"), -1),
        ("fire_unaligned", w("fire_unaligned_penalty"), -1), ("fire_no_ammo", w("fire_no_ammo_penalty"), -1),
        ("fire_aligned", w("fire_aligned_reward"), +1), ("splash", w("splash_fire_reward"), +1),
        ("hook_req", w("hook_required_reward"), +1), ("hook_cost", w("hook_cost"), -1),
        ("hook_enemy", w("hook_enemy_reward"), +1), ("hook_no_ammo", w("hook_no_ammo_reward"), +1),
        ("hook_blind", w("hook_blind_penalty"), -1),
    ],
    "SURVIVAL (per step)": [
        ("tick", w("survival_tick_reward"), +1), ("threat", w("survival_threat_reward"), +1),
        ("low_health", w("survival_low_health_reward"), +1),
    ],
    "EXT channels (per step)": [
        ("damage_prox_aversion", w("damage_prox_aversion"), -1),
        ("offense_rune", w("offense_rune_reward"), +1),
        ("survival_rune", w("survival_rune_reward"), +1),
    ],
    "EPISODE outcome (per episode)": [
        ("win", w("episode_win_reward"), +1), ("survival", w("episode_survival_reward"), +1),
        ("loss×deaths", w("episode_loss_penalty"), -1), ("idle", w("episode_idle_penalty"), -1),
        ("frag_adv", w("frag_advantage_reward"), +1), ("frag_disadv", w("frag_disadvantage_penalty"), -1),
        ("dmg_adv ×margin", w("damage_advantage_reward"), +1),
        ("dmg_disadv ×margin", w("damage_disadvantage_penalty"), -1),
    ],
}

# ── (B) measured per-episode contribution (run-B snapshot, override via env) ──
DMG_MARGIN = float(os.environ.get("EMP_DMG_MARGIN", "-36276"))
EMP = {
    "BASE game":            float(os.environ.get("EMP_BASE", "-54.7")),
    "MEMORY / voxel":       float(os.environ.get("EMP_MEMORY", "-0.007")),
    "THREAT / combat":      float(os.environ.get("EMP_THREAT", "-1.16")),
    "outcome: dmg_disadv":  w("damage_disadvantage_penalty") * DMG_MARGIN,
    "outcome: loss+frag+idle": float(os.environ.get("EMP_OUT_OTHER", "-1.0")),
}

# ── render ──────────────────────────────────────────────────────────────────
fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 11),
                               gridspec_kw={"width_ratios": [1.25, 1]})

labels, vals, colors, ticks, tlabels = [], [], [], [], []
y = 0
for bucket, terms in BUCKETS.items():
    ticks.append(y - 0.5); tlabels.append("")
    for name, weight, sign in terms:
        labels.append(f"{name}")
        signed = weight * sign
        vals.append(signed)
        colors.append("#2e7d32" if sign > 0 else "#c62828")
        y += 1
    # bucket header row (blank gap)
    labels.append(f"── {bucket} ──"); vals.append(0); colors.append("none"); y += 1

ypos = range(len(labels))
axA.barh(list(ypos), vals, color=colors)
axA.set_yticks(list(ypos)); axA.set_yticklabels(labels, fontsize=8)
axA.invert_yaxis()
axA.axvline(0, color="k", lw=0.8)
axA.set_xlabel("configured weight  (green=reward, red=penalty)")
axA.set_title("(A) Designed reward surface — every term to scale", fontsize=11)
axA.set_xlim(-0.8, 2.6)
axA.grid(axis="x", alpha=0.3)

# Panel B: empirical contribution
ekeys = list(EMP.keys()); evals = [EMP[k] for k in ekeys]
ecolors = ["#c62828" if v < 0 else "#2e7d32" for v in evals]
bars = axB.barh(range(len(ekeys)), evals, color=ecolors)
axB.set_yticks(range(len(ekeys))); axB.set_yticklabels(ekeys, fontsize=10)
axB.invert_yaxis(); axB.axvline(0, color="k", lw=0.8)
axB.set_xlabel("mean contribution per episode (reward units)")
axB.set_title("(B) Measured surface — one term dwarfs the rest", fontsize=11)
axB.grid(axis="x", alpha=0.3)
for b, v in zip(bars, evals):
    axB.text(v - (4 if v < 0 else -4), b.get_y() + b.get_height()/2,
             f"{v:.1f}", va="center", ha="right" if v < 0 else "left", fontsize=9)

fig.suptitle("Quake-2 ML bot — full reward enumeration  (48 terms, 8 buckets)  "
             f"| empirical: damage_margin={DMG_MARGIN:,.0f}/ep", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.environ.get("OUT", "/tmp/reward_enumeration.png")
fig.savefig(out, dpi=130)
print("wrote", out)
