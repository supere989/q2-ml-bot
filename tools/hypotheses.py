"""
hypotheses.py — registry of experiments the Karpathy loop will try.

Each hypothesis is a dict of {flag_name: value} overrides passed to
train.ppo. Hypotheses are designed to be cheap to test (a single CLI flag
change) and orthogonal where possible (each tests one variable).

Phases:
  Phase 1: wake up the reward signal (currently +0.00).
  Phase 2: tune for steady reward improvement.
  Phase 3: scale up / refine architecture.
"""

HYPOTHESES = [
    # ─── Phase 1: get reward signal flowing ──────────────────────────────
    {
        "name":       "baseline",
        "hypothesis": "current defaults — establishes reference KPI",
        "overrides":  {},
        "phase":      1,
    },
    {
        "name":       "long_episodes",
        "hypothesis": "max_ep_steps=5000 lets bots actually engage before truncation",
        "overrides":  {"max_ep_steps": 5000},
        "phase":      1,
    },
    {
        "name":       "unlimited_episodes",
        "hypothesis": "no truncation — episodes end only on death",
        "overrides":  {"max_ep_steps": 999999},
        "phase":      1,
    },
    {
        "name":       "high_entropy",
        "hypothesis": "10x entropy bonus forces exploration",
        "overrides":  {"ent_coef": 0.1},
        "phase":      1,
    },
    {
        "name":       "very_high_entropy",
        "hypothesis": "50x entropy bonus — bot must explore",
        "overrides":  {"ent_coef": 0.5},
        "phase":      1,
    },
    {
        "name":       "higher_lr",
        "hypothesis": "lr=1e-3 (3x) for faster early learning",
        "overrides":  {"lr": 1e-3},
        "phase":      1,
    },
    {
        "name":       "lower_lr",
        "hypothesis": "lr=1e-4 (3x lower) for stability",
        "overrides":  {"lr": 1e-4},
        "phase":      1,
    },
    {
        "name":       "more_opponents",
        "hypothesis": "4 bots per server = more combat density",
        "overrides":  {"n_bots_per_server": 4},
        "phase":      1,
    },
    {
        "name":       "more_servers",
        "hypothesis": "scale to 8 servers — RTX 2080 has headroom",
        "overrides":  {"n_servers": 8},
        "phase":      1,
    },

    # ─── Phase 2: PPO tuning ──────────────────────────────────────────────
    {
        "name":       "small_clip",
        "hypothesis": "clip_eps=0.1 for more conservative updates",
        "overrides":  {"clip_eps": 0.1},
        "phase":      2,
    },
    {
        "name":       "large_clip",
        "hypothesis": "clip_eps=0.3 allows bigger policy steps",
        "overrides":  {"clip_eps": 0.3},
        "phase":      2,
    },
    {
        "name":       "more_epochs",
        "hypothesis": "8 PPO epochs per rollout instead of 4",
        "overrides":  {"n_epochs": 8},
        "phase":      2,
    },
    {
        "name":       "longer_rollout",
        "hypothesis": "n_steps=512 for richer rollouts",
        "overrides":  {"n_steps": 512},
        "phase":      2,
    },
    {
        "name":       "shorter_rollout",
        "hypothesis": "n_steps=128 for tighter feedback",
        "overrides":  {"n_steps": 128},
        "phase":      2,
    },
    {
        "name":       "lower_gamma",
        "hypothesis": "gamma=0.95 — care less about distant future",
        "overrides":  {"gamma": 0.95},
        "phase":      2,
    },
    {
        "name":       "higher_gamma",
        "hypothesis": "gamma=0.995 — care more about long-term reward",
        "overrides":  {"gamma": 0.995},
        "phase":      2,
    },
    {
        "name":       "lower_gae_lambda",
        "hypothesis": "gae_lambda=0.9 for less advantage bootstrap noise",
        "overrides":  {"gae_lambda": 0.9},
        "phase":      2,
    },
    {
        "name":       "higher_vf_coef",
        "hypothesis": "vf_coef=1.0 — prioritize value-fn fit",
        "overrides":  {"vf_coef": 1.0},
        "phase":      2,
    },
    {
        "name":       "larger_batch",
        "hypothesis": "batch_size=1024 for more stable gradients",
        "overrides":  {"batch_size": 1024},
        "phase":      2,
    },
    {
        "name":       "smaller_batch",
        "hypothesis": "batch_size=256 for more update steps",
        "overrides":  {"batch_size": 256},
        "phase":      2,
    },
]


def by_name(name: str):
    for h in HYPOTHESES:
        if h["name"] == name:
            return h
    return None


def by_phase(phase: int):
    return [h for h in HYPOTHESES if h["phase"] == phase]
