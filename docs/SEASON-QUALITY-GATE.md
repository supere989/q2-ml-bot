# Distributed Training Season Quality Gate

Promotion is based on completed training seasons, not elapsed wall-clock time.
A season is a bounded policy interval with an explicit start/end version and a
JSON evidence report. Faster hardware may finish a season sooner; interrupted
or under-covered work does not pass merely because it ran for a long time.

Default per-season requirements:

- at least 100 PPO generations and 1,000,000 environment steps;
- at least 100 completed episodes on every approved map;
- zero unrecovered failures, duplicate optimizer updates, or non-finite updates;
- p95 approximate KL no greater than 0.03 and p95 clip fraction no greater than 0.20;
- distributed SPS at least 1.25 times the measured local baseline;
- explicit no-regression results for aim, reward, KD, and lattice memory;
- a passing CPU deterministic audit.

Three uniquely named successful seasons are required by default. Thresholds
are CLI-configurable but must be fixed before a promotion campaign starts.

Example report:

```json
{
  "season_id": "lattice-s01",
  "start_policy_version": 40000000,
  "end_policy_version": 41200000,
  "generations": 120,
  "env_steps": 1200000,
  "maps": {
    "mltrain_00005208": {"episodes": 140},
    "mltrain_00006144": {"episodes": 133}
  },
  "recovery": {"unrecovered_failures": 0, "duplicate_optimizer_updates": 0},
  "stability": {
    "nonfinite_updates": 0,
    "approx_kl_p95": 0.012,
    "clip_fraction_p95": 0.08
  },
  "performance": {"distributed_sps": 60.0, "baseline_sps": 20.0},
  "regressions": {
    "aim": false,
    "reward": false,
    "kd": false,
    "lattice_memory": false
  },
  "cpu_deterministic_audit": true
}
```

Evaluate a promotion campaign:

```bash
python tools/season_quality_gate.py \
  season-01.json season-02.json season-03.json
```

The command exits zero only when the campaign passes and emits a JSON decision
containing every per-season failure.
