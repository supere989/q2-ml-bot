"""Composition root for the new multires trainer generation.

It intentionally does not call the legacy PPO resume path.  B4 supplies
already-admitted 198/24/16/60 blocks and runtime evidence; this runtime applies
seeded advisory dropout, the fresh model graph, causal rewards, and attested
same-generation checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from harness.multires_contract import (
    GUIDES,
    OBS_DIM,
    GuideDropoutConfig,
    GuideDropoutIdentity,
    apply_seeded_guide_dropout,
    pack_policy_vector,
)
from harness.multires_lineage import (
    MultiresCheckpointManifest,
    load_attested_checkpoint,
    save_attested_checkpoint,
)
from harness.multires_reward import (
    CausalRewardConfig,
    CausalRewardFrame,
    CausalRewardReducer,
    CausalRewardResult,
)
from harness.multires_runtime import (
    ValidatedMultiresRuntime,
    validate_runtime_evidence,
)
from harness.multires_training_config import MultiresTrainingConfiguration
from models.multires_policy import MultiresQ2BotPolicy
from .multires_ppo import MultiresPPOConfig


@dataclass
class MultiresTrainerRuntime:
    policy: MultiresQ2BotPolicy
    runtime: ValidatedMultiresRuntime
    reward_config: CausalRewardConfig
    guide_dropout: GuideDropoutConfig
    ppo_config: MultiresPPOConfig
    training_config: MultiresTrainingConfiguration
    initialization: str
    lineage_root_sha256: Optional[str] = None
    reward_reducers: dict[str, CausalRewardReducer] = field(default_factory=dict)

    @classmethod
    def fresh(
        cls,
        runtime_evidence: Mapping[str, Any],
        *,
        expected_atlas_sha256: str,
        seed: int,
        device: torch.device = torch.device("cpu"),
        initialization: str = "random",
        reward_config: CausalRewardConfig = CausalRewardConfig(),
        guide_dropout: GuideDropoutConfig = GuideDropoutConfig(),
        ppo_config: MultiresPPOConfig = MultiresPPOConfig(),
    ) -> "MultiresTrainerRuntime":
        if initialization != "random":
            raise ValueError(
                "fresh runtime is random-only; new-schema BC requires an "
                "explicitly attested projection path"
            )
        if not runtime_evidence.get("runtime_manifest_sha256"):
            raise ValueError("multires trainer requires a sealed runtime manifest")
        runtime = validate_runtime_evidence(
            runtime_evidence, expected_atlas_sha256=expected_atlas_sha256
        )
        reward_config.validate()
        guide_dropout.validate()
        ppo_config.validate()
        training_config = MultiresTrainingConfiguration.create(
            reward=reward_config,
            guide_dropout=guide_dropout,
            ppo=ppo_config,
        )
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        policy = MultiresQ2BotPolicy().to(device)
        return cls(
            policy=policy,
            runtime=runtime,
            reward_config=reward_config,
            guide_dropout=guide_dropout,
            ppo_config=ppo_config,
            training_config=training_config,
            initialization=initialization,
        )

    @classmethod
    def resume(
        cls,
        checkpoint: Path,
        runtime_evidence: Mapping[str, Any],
        *,
        expected_atlas_sha256: str,
        device: torch.device = torch.device("cpu"),
        optimizer_factory: Callable[[Any], Any],
        reward_config: CausalRewardConfig = CausalRewardConfig(),
        guide_dropout: GuideDropoutConfig = GuideDropoutConfig(),
        ppo_config: MultiresPPOConfig = MultiresPPOConfig(),
        expected_lineage_root_sha256: Optional[str] = None,
    ) -> tuple["MultiresTrainerRuntime", Any, MultiresCheckpointManifest]:
        if not runtime_evidence.get("runtime_manifest_sha256"):
            raise ValueError("multires trainer requires a sealed runtime manifest")
        runtime = validate_runtime_evidence(
            runtime_evidence, expected_atlas_sha256=expected_atlas_sha256
        )
        reward_config.validate()
        guide_dropout.validate()
        ppo_config.validate()
        training_config = MultiresTrainingConfiguration.create(
            reward=reward_config,
            guide_dropout=guide_dropout,
            ppo=ppo_config,
        )
        policy = MultiresQ2BotPolicy().to(device)
        optimizer = optimizer_factory(policy.parameters())
        manifest = load_attested_checkpoint(
            checkpoint,
            policy,
            expected_atlas_sha256=runtime.atlas_sha256,
            expected_runtime_manifest_sha256=runtime.runtime_manifest_sha256,
            expected_training_config=training_config,
            optimizer=optimizer,
            expected_lineage_root_sha256=expected_lineage_root_sha256,
            map_location=device,
        )
        restored = cls(
            policy=policy,
            runtime=runtime,
            reward_config=reward_config,
            guide_dropout=guide_dropout,
            ppo_config=ppo_config,
            training_config=training_config,
            initialization=manifest.initialization,
            lineage_root_sha256=manifest.lineage_root_sha256,
        )
        return restored, optimizer, manifest

    def prepare_observation(
        self,
        factual: Sequence[float],
        dyn: Sequence[float],
        recovery: Sequence[float],
        guides: Sequence[float],
        *,
        dropout_identity: GuideDropoutIdentity,
        training: bool = True,
    ) -> tuple[np.ndarray, tuple[bool, ...]]:
        if training:
            dropout = apply_seeded_guide_dropout(
                guides, dropout_identity, self.guide_dropout
            )
            guide_values = dropout.guides
            dropped = dropout.dropped_candidates
        else:
            guide_values = np.asarray(guides, dtype=np.float32)
            dropped = (False, False, False, False)
        return pack_policy_vector(
            factual, dyn, recovery, guide_values
        ), dropped

    def prepare_policy_vector(
        self,
        vector: Sequence[float],
        *,
        dropout_identity: GuideDropoutIdentity,
        training: bool = True,
    ) -> tuple[np.ndarray, tuple[bool, ...]]:
        """Apply advisory dropout without rebuilding already-attested blocks."""
        result = np.asarray(vector, dtype=np.float32)
        if result.shape != (OBS_DIM,) or not np.isfinite(result).all():
            raise ValueError(f"policy vector must be finite shape ({OBS_DIM},)")
        result = result.copy()
        if not training:
            return result, (False, False, False, False)
        dropout = apply_seeded_guide_dropout(
            result[GUIDES.slice], dropout_identity, self.guide_dropout
        )
        result[GUIDES.slice] = dropout.guides
        return result, dropout.dropped_candidates

    def reward(
        self, client_id: str, frame: CausalRewardFrame
    ) -> CausalRewardResult:
        """Reduce one admitted client stream without cross-client state bleed."""
        if not client_id:
            raise ValueError("reward attribution requires a stable client_id")
        reducer = self.reward_reducers.get(client_id)
        if reducer is None:
            reducer = CausalRewardReducer(self.reward_config)
            self.reward_reducers[client_id] = reducer
        return reducer.step(frame)

    def reset_reward_stream(self, client_id: str) -> None:
        """Discard all per-life causal state at a transport/map boundary."""
        if not client_id:
            raise ValueError("reward reset requires a stable client_id")
        self.reward_reducers.pop(client_id, None)

    def checkpoint(
        self,
        path: Path,
        *,
        training_step: int,
        optimizer: Any,
    ) -> MultiresCheckpointManifest:
        manifest = save_attested_checkpoint(
            path,
            self.policy,
            atlas_sha256=self.runtime.atlas_sha256,
            runtime_manifest_sha256=self.runtime.runtime_manifest_sha256,
            training_config=self.training_config,
            initialization=self.initialization,
            training_step=training_step,
            optimizer=optimizer,
            lineage_root_sha256=self.lineage_root_sha256,
        )
        self.lineage_root_sha256 = manifest.lineage_root_sha256
        return manifest
