from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harness.multires_reward import CausalRewardConfig
from tests.b5_evidence_fixtures import SOURCE, write_inputs
import tools.run_multires_pretraining_campaign as campaign_tool
from tools.run_multires_pretraining_campaign import (
    CampaignContext,
    _evaluate,
    run_campaign,
)
from tools.run_multires_pretraining_validation import (
    CAMPAIGN_SCHEMA,
    REQUIRED_CAMPAIGN_MODES,
    _validate_campaign_results,
    canonical_sha256,
)
from train.multires_one_run import _state_sha256
from train.multires_runtime import MultiresTrainerRuntime


CATALOG = "c" * 64


class _AtlasRuntime:
    def guide_features(self, position, yaw, map_epoch, beliefs):
        assert map_epoch == 1
        assert beliefs
        rows = np.zeros((4, 15), dtype=np.float32)
        for index in range(4):
            rows[index, :7] = (
                1.0 - index * 0.1,
                (float(position[1]) / 4096.0) + index * 0.05,
                float(position[2]) / 4096.0,
                0.1 + index * 0.1,
                0.0,
                1.0,
                1.0,
            )
            rows[index, 7 + index] = 1.0
        return rows.reshape(-1).tolist()


def _context(paths: dict[str, Path], *, seed: int = 17) -> CampaignContext:
    runtime_document = json.loads(paths["runtime_manifest"].read_bytes())
    atlas_sha256 = campaign_tool._sha256(paths["atlas"])
    runtime = MultiresTrainerRuntime.fresh(
        runtime_document,
        expected_atlas_sha256=atlas_sha256,
        active_atlas_sha256=atlas_sha256,
        expected_atlas_catalog_sha256=CATALOG,
        seed=seed,
        reward_config=CausalRewardConfig(),
    )
    optimizer = torch.optim.Adam(runtime.policy.parameters(), lr=3e-4)
    admission = SimpleNamespace(
        args=SimpleNamespace(
            map_name="test_arena",
            expected_atlas_sha256=atlas_sha256,
            expected_atlas_catalog_sha256=CATALOG,
        ),
        atlas_sha256=atlas_sha256,
        runtime_manifest_sha256=runtime.runtime.runtime_manifest_sha256,
        objective_identity_sha256="ab" * 32,
    )
    return CampaignContext(
        admission=admission,
        runtime=runtime,
        optimizer=optimizer,
        provider=SimpleNamespace(atlas_runtime=_AtlasRuntime()),
        objectives=(
            (10, (256.0, 0.0, 64.0)),
            (11, (512.0, 128.0, 96.0)),
        ),
        policy_state_before=_state_sha256(runtime.policy.state_dict()),
        optimizer_state_before=_state_sha256(optimizer.state_dict()),
    )


@pytest.mark.parametrize("mode", REQUIRED_CAMPAIGN_MODES)
def test_each_repository_evaluator_executes_real_policy_reward_and_season(
    tmp_path: Path, mode: str,
) -> None:
    paths = write_inputs(tmp_path)
    context = _context(paths)
    results, trajectory, season = _evaluate(context, mode, 8)
    _validate_campaign_results(mode, results)
    assert season["schema"] == "multires-atlas-season-v1"
    assert season["accepted_transitions"] == 8
    assert season["transport"] == {
        "command_echo_match_rate": 1.0,
        "state_resyncs": 0,
    }
    if mode in {"guide_on", "guide_off"}:
        assert "hidden_teacher_field_violations" not in results
    assert season["privilege"] == {
        "scope": "not-measured-offline-no-public-conduit",
        "upstream_evidence_required": "sealed-b4-real-public-datagram-audit-v1",
    }
    assert len(trajectory) == 64
    int(trajectory, 16)
    assert _state_sha256(context.runtime.policy.state_dict()) == context.policy_state_before
    assert _state_sha256(context.optimizer.state_dict()) == context.optimizer_state_before


def test_guide_pair_uses_same_rust_scenario_and_real_seeded_dropout(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    on, _on_trace, on_season = _evaluate(_context(paths), "guide_on", 32)
    off, _off_trace, off_season = _evaluate(_context(paths), "guide_off", 32)
    assert on["scenario_identity_sha256"] == off["scenario_identity_sha256"]
    assert 0 <= on["guide_dropout_samples"] <= 32 * 4
    assert on["guide_nonzero_samples"] > 0
    assert off["guide_dropout_samples"] == 32 * 4
    assert off["guide_nonzero_samples"] == 0
    assert round(on_season["guides"]["candidate_drop_rate"] * 32 * 4) == on[
        "guide_dropout_samples"
    ]
    assert off_season["guides"]["candidate_drop_rate"] == 1.0


def test_no_update_probe_observes_optimizer_and_backward_paths(tmp_path: Path) -> None:
    context = _context(write_inputs(tmp_path))
    with campaign_tool._NoUpdateProbe(context) as probe:
        context.optimizer.step()
        next(context.runtime.policy.parameters()).sum().backward()
    assert probe.counters["optimizer_steps"] == 1
    assert probe.counters["backward_parameter_gradients"] == 1
    with pytest.raises(campaign_tool.CampaignError, match="mutation path"):
        probe.validate()


@pytest.mark.parametrize("mode", REQUIRED_CAMPAIGN_MODES)
def test_run_campaign_derives_canonical_evidence_without_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    paths = write_inputs(tmp_path)
    q2ded = tmp_path / "q2ded"
    client = tmp_path / "q2-client"
    objectives = tmp_path / "objectives.json"
    atlas_catalog = tmp_path / "atlas-catalog.json"
    for path in (q2ded, client):
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    objectives.write_text("{}\n", encoding="ascii")
    atlas_catalog.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(campaign_tool, "_git_identity", lambda *_args: None)
    args = SimpleNamespace(
        protocol=CAMPAIGN_SCHEMA,
        campaign=mode,
        replicate=0,
        seed=17,
        game_seed=23,
        transition_count=8,
        repo_commit=SOURCE["commit"],
        repo_tree=SOURCE["tree"],
        runtime_manifest=paths["runtime_manifest"],
        checkpoint=paths["checkpoint"],
        training_manifest=paths["training_manifest"],
        bundle_manifest=paths["bundle_manifest"],
        atlas_bin=paths["atlas"],
        atlas_catalog=atlas_catalog,
        expected_atlas_catalog_sha256=CATALOG,
        q2ded=q2ded,
        client_binary=client,
        runtime_root=tmp_path,
        objectives=objectives,
        no_update=True,
        output=tmp_path / "out.json",
    )
    monkeypatch.setattr(campaign_tool, "build_context", lambda _args: _context(paths))
    evidence = run_campaign(args)
    assert evidence["status"] == "passed"
    assert evidence["counters"] == campaign_tool.ZERO_COUNTERS
    assert evidence["season_report"]["accepted_transitions"] == 8
    assert evidence["season_report_sha256"] == canonical_sha256(
        evidence["season_report"]
    )
    assert evidence["result_sha256"] == campaign_tool.campaign_result_sha256(evidence)
