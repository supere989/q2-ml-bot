# Lattice role gates — 2026-07-23

Status: implemented and verified. The lattice subsystem now matches the
documented architecture: one physical store, per-role channels, gates at
the consumption points. All defaults preserve prior behavior; no run
configuration changes are required.

## Problem

The gates that existed were tied to storage, not consumption.
`Q2_SESSION_MEMORY=0` stopped deposits, policy-input features, and
memory-derived reward together, so the design's core ablation ("policy sees
lattice but is not shaped by it", and vice versa) could not be run. The
aux direction loss tracked whatever producer owned obs slots 5:9 with no
attribution. Restored checkpoints silently shadowed generator priors.
External coach directives wrote into the store with no attestation.

## Changes (harness/spatial.py, train/ppo.py)

- **Storage vs consumption decoupled.** Deposits stay live under every
  ablation so runs still collect scoring data. New consumption gates:
  - `Q2_LATTICE_OBS` (default: follows `Q2_SESSION_MEMORY`) — zeroes the
    entire 24-float memory tail at the policy-input boundary.
  - `Q2_LATTICE_REWARD` (default: follows `Q2_SESSION_MEMORY`) — removes
    only the lattice-derived reward terms (engagement/opportunity pull,
    threat penalty, death aversion, self-fire penalty, camp penalty, and
    the lattice-guided hook correction, including its blind-hook penalty
    asymmetry). Gameplay reward terms are untouched.
  - `Q2_LATTICE_IMMEDIATE` (default 1) — zeroes only the immediate
    engagement slice (5:9) regardless of producer (thermal-hot or
    persistent fallback); the clean ablation switch for that channel.
  - `Q2_LATTICE_DIRECTIVES` (default 1) — gates `apply_directive`;
    rejected directives log a warning, applied ones log map/action/cell.
- **Reward computation reads the ungated internal feature vector**
  (`_memory_features_internal`); `memory_features` is now the public gated
  view used at the obs boundary. An obs ablation can no longer silently
  change reward semantics mid-run.
- **Restore re-merges priors.** `load_lattice_state` no longer marks
  restored maps as preloaded, so generator priors max-merge (idempotent)
  into restored cells on the next reset. A resumed run now trains on the
  same prior substrate as a fresh one.
- **Run config provenance.** `train/ppo.py` records all seven lattice
  flags in the distributed config payload (this changes
  `distributed_config_hash`, intentionally).

Backward compatibility: `Q2_SESSION_MEMORY=0` still disables everything by
default (the consumption gates follow it unless overridden).

## Validation

- `tests/test_lattice_gates.py` (new, 7 tests): obs gate zeroes public
  tail while storage accumulates; reward gate removes lattice terms and
  keeps gameplay terms; immediate gate zeroes slice 5:9 under a hot
  thermal track; legacy `Q2_SESSION_MEMORY=0` semantics; override
  independence; restore prior re-merge; directive gate + logging.
- Full suite: 132 passed, 11 skipped (torch-dependent skips; torch is not
  installed on this host).
- Live gate-integrity probe against the local conduit (two headless
  clients, `reset()` + 5 `step_vector`s): `Q2_LATTICE_OBS=1` tail sum
  1.5, `Q2_LATTICE_OBS=0` tail sum exactly 0.0.

## What this unblocks

The ablation matrix the design has always called for, now runnable as
config rather than code:

| run | env | question |
|---|---|---|
| all-on | (defaults) | baseline |
| reward-off | `Q2_LATTICE_REWARD=0` | does lattice shaping change behavior? |
| obs-off | `Q2_LATTICE_OBS=0` | does the policy use the memory tail? |
| immediate-off | `Q2_LATTICE_IMMEDIATE=0` | is thermal pursuit carrying combat? |
| direction-off | `lattice_direction_coef=0` | does the aux loss help or harm? |

Match seeds per arm (`--seed N --game_seed N --deterministic 1`,
`Q2_ML_ASYNC=0`) and compare hit/kill/contact metrics plus
`evaluate_lattice.py --require_mean_cosine 0.25`. Training runs on the
WSL box; do not run these on the workstation.
