# Survival & success information corpus — arena deathmatch — 2026-07-24

Status: ratified design; Phase 1 implemented in this change set.

This document defines the complete corpus of information the lattice game
model must carry for an agent to **survive** (not die) and **succeed** (win
the frag race) in Quake II arena deathmatch with the Lithium ruleset
(grapple, runes). It follows the project's three-timescale memory doctrine
and its privilege taxonomy: every item names its authority source, its
privilege class, and its delivery status.

Privilege classes (per the multires Atlas taxonomy):
- **policy-factual** — knowledge a fair player could have; may enter the
  policy input (in a future input-v2 lineage) and reward shaping now.
- **advisory ablatable** — lattice guidance; consumption-gated, removable
  by design (see docs/LATTICE-ROLE-GATES-2026-07-23.md).
- **teacher-only** — oracle state used for scoring/curriculum, packed on a
  physically separate path, asserted absent on the public conduit.

Authority boundaries (unchanged, non-negotiable): fire authorization and
exact aim geometry come only from current-frame engine visibility/exposure;
lattice cells never authorize actions; unknown is explicit, never guessed.

## Tier 0 — static map truth (Atlas role)

| information | authority | class | status |
|---|---|---|---|
| geometry/solidity priors | generator/physics oracle sidecars | advisory | exists (`.lattice.json`) |
| hazard regions, TYPED (lava/void/crush/fall-line) | map contents + oracle | advisory | partial: untyped `danger` bounds only → P3 (Atlas bitplanes) |
| item spawn table + respawn periods | entity lump / generator | advisory | exists (`.routes.json`) |
| route graph (risk-weighted room connectivity) | generator | advisory | exists |
| hook zones (anchor→landing, required flags) | map sidecar | advisory | exists |

## Tier 1 — persistent personal memory (Dyn role)

| information | authority | class | status |
|---|---|---|---|
| per-cell combat history (engagement/threat/opportunity/self_fire/deaths) | own telemetry deposits | advisory | exists |
| **typed hazard events** (hazard_damage, hazard_deaths) | MOD classification of own damage/deaths | advisory | **added in P1** |
| **death attribution hazard-vs-combat** | `last_death_mod` on wire | advisory | **added in P1** (threat de-conflated) |
| item availability beliefs + confidence | respawn clocks + own observed pickups | advisory | partial: enemy pickups invisible by design; oracle upgrade → P2 |
| engagement win-margin (survivability projection) | self state + measured DPS | advisory | exists (obs slots 21–23) |

## Tier 2 — ephemeral perception (thermal role)

| information | authority | class | status |
|---|---|---|---|
| exact target solution (eye-to-damage-point, exposure) | engine traces, current frame | policy-factual | exists |
| pursuit continuity (5-tick cooling heat, never fire-authorizing) | thermal overlay | policy-factual | exists |
| **per-target hit attribution** (edict+epoch of last hit/kill) | server combat log | policy-factual | **added in P1** |
| **self-exposure** (max exposure any live enemy has of me) | symmetric engine traces | policy-factual | **added in P1** |
| **fight decision bias** (scalar ∈ [-1,1]: take this fight?) | derived: win-margin projection + exposure asymmetry + typed-hazard proximity + outnumbered + escape readiness | advisory ablatable | **added 2026-07-24** (reward/deposit path only; never masks or alters actions) |

### Fight decision bias (advisory, 2026-07-24)

`fight_bias = clip(w_margin·margin + w_exposure·(my_exposure − self_exposure)
− w_hazard·hazard − w_outnumbered·(count−1) + w_escape·escape, −1, 1)`,
computed per tick in `VoxelSpatialReward.update` (active while alive and on
the death tick itself; corpse frames read 0.0). It exists because the
engagement pull was pushing the policy into losing fights (matrix: 80
deaths all-on vs 34 without). It acts ONLY through:

- reward shaping (gated by `lattice_reward_enabled`): engaging at
  bias < −0.2 costs `R_BAD_FIGHT_PENALTY` (0.010) × |bias|; losing sight of
  the enemy that tick at bias < −0.2 pays `R_DISENGAGE_REWARD` (0.006) ×
  |bias|; engaging at bias > +0.3 pays `R_GOOD_FIGHT_REWARD` (0.004) × bias.
- typed decision-quality deposits: `bad_fight_taken` (death at bias < −0.2,
  at the death cell) and `good_disengage` (rewarded disengage). These feed
  the EXISTING threat read channel at `R_BAD_FIGHT_THREAT_WEIGHT` (0.5) —
  the 24-float policy tail layout is frozen; no new read channel.

Env knobs: `Q2_BIAS_W_MARGIN` (0.5), `Q2_BIAS_W_EXPOSURE` (0.3),
`Q2_BIAS_W_HAZARD` (0.4), `Q2_BIAS_W_OUTNUMBERED` (0.15),
`Q2_BIAS_W_ESCAPE` (0.1), `R_BAD_FIGHT_PENALTY`, `R_DISENGAGE_REWARD`,
`R_GOOD_FIGHT_REWARD`, `R_BAD_FIGHT_THREAT_WEIGHT`.

## Tier 3 — game frame

| information | authority | class | status |
|---|---|---|---|
| self health/armor/ammo/weapon/velocity | engine | policy-factual | exists |
| **score_self / score_leader / time_remaining** | scoreboard | policy-factual | **added in P1** |
| spawn protection / invincibility | engine | policy-factual | exists (fire-gate pad bits) |
| rune state | engine | policy-factual | exists (EXT_OBS) |
| inbound damage vector + recency | engine | policy-factual | exists (EXT_OBS) |

## Teacher-only (oracle scoring — never policy)

| information | authority | class | status |
|---|---|---|---|
| exact enemy item pickups (who took what, when) | engine item touch | teacher-only | → P2 |
| full enemy inventories/readiness | engine | teacher-only | → P2 |
| item-control skill score (economy dominance) | derived from above | teacher-only | → P2 |

## What the corpus says about the current agent (honest read)

- Survival was previously underserved: deaths/damage were recorded without
  cause, so the model blamed *locations* for hazards it could not type, and
  the combat threat channel absorbed environmental noise. P1 fixes the
  attribution; P3 will add static typed hazard maps so the agent can
  anticipate rather than only remember.
- Success lacks economy awareness: the agent cannot know the item clock
  state of things it didn't see taken. That is *correct* for fairness
  (policy side) and exactly what the teacher-only channel is for
  (curriculum side) — P2 closes the loop by *scoring* economy skill
  without leaking the answers.
- Perception must stay honest: nothing in this corpus authorizes firing
  through memory, and no belief is represented as fact. Where confidence
  is low, the field says so (explicit-unknown law).

## Phase roadmap

- **P1 (this change set)**: wire v5 survival pack — MOD-typed damage/death
  attribution, per-target hit attribution, self-exposure, score frame;
  hazard channels in the lattice with threat de-conflation; thermal
  per-target track clearing. Reward/attribution only; the 219-dim policy
  input is untouched.
- **P2**: teacher-only item-control channel + policy-input v2 lineage
  decision (hazard/exposure/score perception dims; new checkpoint
  generation, not a resume).
- **P3**: Atlas static typed hazard bitplanes per
  docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md; P1's typed events
  become its Dyn input.
