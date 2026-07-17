"""Canonical, attested configuration for the multires training generation.

Reward coefficients, guide dropout, and PPO hyperparameters are semantic
runtime inputs.  A checkpoint trained under different values is a different
lineage even when its tensor shapes happen to match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping


TRAINING_CONFIG_SCHEMA = "q2-multires-training-config-v1"


class TrainingConfigError(ValueError):
    """Raised when a training configuration is not canonical JSON data."""


def _normalize(value: Any, label: str) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, f"{label}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item, label) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingConfigError(f"{label} contains NaN or infinity")
        return value
    raise TrainingConfigError(
        f"{label} contains unsupported value {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise TrainingConfigError("training configuration is not canonical JSON") from error


@dataclass(frozen=True)
class MultiresTrainingConfiguration:
    schema: str
    reward: Mapping[str, Any]
    guide_dropout: Mapping[str, Any]
    ppo: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        reward: Any,
        guide_dropout: Any,
        ppo: Any,
    ) -> "MultiresTrainingConfiguration":
        value = cls(
            schema=TRAINING_CONFIG_SCHEMA,
            reward=_normalize(reward, "reward"),
            guide_dropout=_normalize(guide_dropout, "guide_dropout"),
            ppo=_normalize(ppo, "ppo"),
        )
        # Exercise canonical serialization at the admission boundary.
        value.to_json()
        return value

    @classmethod
    def from_json(cls, value: str) -> "MultiresTrainingConfiguration":
        if not isinstance(value, str) or not value:
            raise TrainingConfigError("training configuration JSON is missing")
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise TrainingConfigError("training configuration JSON is invalid") from error
        if not isinstance(decoded, Mapping) or set(decoded) != {
            "schema", "reward", "guide_dropout", "ppo"
        }:
            raise TrainingConfigError("training configuration envelope differs")
        if decoded["schema"] != TRAINING_CONFIG_SCHEMA:
            raise TrainingConfigError("training configuration schema differs")
        if not all(
            isinstance(decoded[name], Mapping)
            for name in ("reward", "guide_dropout", "ppo")
        ):
            raise TrainingConfigError("training configuration sections must be mappings")
        result = cls.create(
            reward=decoded["reward"],
            guide_dropout=decoded["guide_dropout"],
            ppo=decoded["ppo"],
        )
        if result.to_json() != value:
            raise TrainingConfigError("training configuration JSON is not canonical")
        return result

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reward": dict(self.reward),
            "guide_dropout": dict(self.guide_dropout),
            "ppo": dict(self.ppo),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_mapping())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
