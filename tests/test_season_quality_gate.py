from harness.season_quality_gate import SeasonGateConfig, evaluate_promotion, evaluate_season


def _report(season_id="season-1"):
    return {
        "season_id": season_id,
        "start_policy_version": 40_000_000,
        "end_policy_version": 41_200_000,
        "generations": 120,
        "env_steps": 1_200_000,
        "maps": {"map-a": {"episodes": 150}, "map-b": {"episodes": 125}},
        "recovery": {"unrecovered_failures": 0, "duplicate_optimizer_updates": 0},
        "stability": {
            "nonfinite_updates": 0,
            "approx_kl_p95": 0.012,
            "clip_fraction_p95": 0.08,
        },
        "performance": {"distributed_sps": 60.0, "baseline_sps": 20.0},
        "regressions": {
            "aim": False,
            "reward": False,
            "kd": False,
            "lattice_memory": False,
        },
        "cpu_deterministic_audit": True,
    }


def test_season_passes_by_evidence_not_elapsed_time():
    decision = evaluate_season(_report(), SeasonGateConfig())
    assert decision["passed"]
    assert decision["speedup"] == 3.0


def test_season_fails_coverage_recovery_and_regression_gates():
    report = _report()
    report["maps"]["map-b"]["episodes"] = 2
    report["recovery"]["unrecovered_failures"] = 1
    report["regressions"]["aim"] = True
    report["cpu_deterministic_audit"] = False
    decision = evaluate_season(report, SeasonGateConfig())
    assert not decision["passed"]
    assert any("map map-b" in failure for failure in decision["failures"])
    assert any("unrecovered" in failure for failure in decision["failures"])
    assert any("aim" in failure for failure in decision["failures"])


def test_promotion_requires_unique_successful_seasons():
    config = SeasonGateConfig(min_successful_seasons=3)
    passed = evaluate_promotion(
        [_report("season-1"), _report("season-2"), _report("season-3")], config
    )
    assert passed["passed"]
    failed = evaluate_promotion([_report("same"), _report("same")], config)
    assert not failed["passed"]
    assert failed["successful_seasons"] == 2
