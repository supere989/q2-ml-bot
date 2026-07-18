"""Fail-closed checkpoint and runtime lineage for the multires policy.

Raw legacy state dictionaries are intentionally not recognized.  A resumable
checkpoint must carry the exact 298-feature generation, categorical action
cardinalities, model tensor schema, immutable Atlas-catalog digest, and
runtime-manifest digest.  Active per-map Atlas identity remains frame evidence;
it is not the policy lineage identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

from .multires_contract import (
    ACTION_DIM,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    HOOK_CLASSES,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
    WEAPON_CLASSES,
)
from .multires_training_config import MultiresTrainingConfiguration


CHECKPOINT_FORMAT = "q2-multires-attested-checkpoint-v2"
MODEL_ARCHITECTURE = "models.multires_policy.MultiresQ2BotPolicy"
ALLOWED_INITIALIZATIONS = frozenset(("random", "new-schema-bc"))
_SHA256_CHARS = frozenset("0123456789abcdef")


class LineageError(ValueError):
    """Raised before state mutation when lineage cannot be proven."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _SHA256_CHARS for character in value)


def _tensor_schema(state: Mapping[str, Any]) -> tuple[str, tuple[dict[str, Any], ...]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(state):
        value = state[name]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or dtype is None:
            raise LineageError(f"state entry {name!r} is not tensor-like")
        if (
            (value.is_floating_point() or value.is_complex())
            and not bool(value.isfinite().all())
        ):
            raise LineageError(f"state entry {name!r} contains a non-finite value")
        rows.append({
            "name": str(name),
            "shape": [int(dimension) for dimension in shape],
            "dtype": str(dtype),
        })
    if not rows:
        raise LineageError("policy state schema is empty")
    encoded = _canonical_json(rows)
    return hashlib.sha256(encoded).hexdigest(), tuple(rows)


def _optimizer_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LineageError("optimizer configuration contains a non-finite value")
        return value
    if isinstance(value, (tuple, list)):
        return [_optimizer_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(name): _optimizer_value(item)
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    shape = getattr(value, "shape", None)
    if shape is not None and getattr(value, "numel", lambda: 0)() == 1:
        return _optimizer_value(value.item())
    raise LineageError(
        f"unsupported optimizer configuration value {type(value).__name__}"
    )


def optimizer_identity(
    optimizer: Any, policy: Any
) -> tuple[str, dict[str, Any]]:
    """Bind optimizer class, hyperparameters, and exact policy ownership."""
    if optimizer is None:
        raise LineageError("resumable multires checkpoints require an optimizer")
    policy_parameters = list(policy.parameters())
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    if (
        len(optimizer_parameters) != len(policy_parameters)
        or len({id(parameter) for parameter in optimizer_parameters})
        != len(optimizer_parameters)
        or {id(parameter) for parameter in optimizer_parameters}
        != {id(parameter) for parameter in policy_parameters}
    ):
        raise LineageError(
            "optimizer must own every fresh multires policy parameter exactly once"
        )
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            "parameter_count": len(group["params"]),
            "configuration": _optimizer_value({
                name: value for name, value in group.items() if name != "params"
            }),
        })
    descriptor = {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": groups,
    }
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest(), descriptor


def _optimizer_state_identity(optimizer: Any, state: Mapping[str, Any]) -> str:
    if set(state) != {"state", "param_groups"}:
        raise LineageError("optimizer state has an unexpected envelope")
    groups = state["param_groups"]
    slots = state["state"]
    if not isinstance(groups, list) or not isinstance(slots, Mapping):
        raise LineageError("optimizer state groups/slots are malformed")
    parameter_ids: list[Any] = []
    descriptors = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise LineageError("optimizer parameter group is malformed")
        parameter_ids.extend(group["params"])
        descriptors.append({
            "parameter_count": len(group["params"]),
            "configuration": _optimizer_value({
                name: value for name, value in group.items() if name != "params"
            }),
        })
    if len(set(parameter_ids)) != len(parameter_ids):
        raise LineageError("optimizer state repeats a parameter identifier")
    if not set(slots).issubset(set(parameter_ids)):
        raise LineageError("optimizer state contains an unbound parameter slot")

    def check_finite(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_finite(item)
        elif hasattr(value, "is_floating_point") and (
            value.is_floating_point() or value.is_complex()
        ) and not bool(value.isfinite().all()):
            raise LineageError("optimizer state contains a non-finite tensor")

    check_finite(slots)
    descriptor = {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": descriptors,
    }
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


@dataclass(frozen=True)
class MultiresCheckpointManifest:
    checkpoint_format: str
    policy_generation: str
    feature_schema: str
    feature_schema_sha256: str
    observation_dim: int
    action_dim: int
    posture_classes: int
    fire_classes: int
    hook_classes: int
    weapon_classes: int
    architecture: str
    state_schema_sha256: str
    optimizer_identity_sha256: str
    atlas_catalog_sha256: str
    runtime_manifest_sha256: str
    training_config_json: str
    training_config_sha256: str
    initialization: str
    lineage_root_sha256: str
    training_step: int

    @classmethod
    def create(
        cls,
        *,
        state_schema_sha256: str,
        optimizer_identity_sha256: str,
        atlas_catalog_sha256: str,
        runtime_manifest_sha256: str,
        training_config: MultiresTrainingConfiguration,
        initialization: str,
        training_step: int,
        lineage_root_sha256: Optional[str] = None,
    ) -> "MultiresCheckpointManifest":
        if initialization not in ALLOWED_INITIALIZATIONS:
            raise LineageError(
                "initialization must be random or explicitly projected new-schema BC"
            )
        if training_step < 0:
            raise LineageError("training_step must be nonnegative")
        for name, digest in (
            ("state_schema_sha256", state_schema_sha256),
            ("optimizer_identity_sha256", optimizer_identity_sha256),
            ("atlas_catalog_sha256", atlas_catalog_sha256),
            ("runtime_manifest_sha256", runtime_manifest_sha256),
            ("training_config_sha256", training_config.sha256),
        ):
            if not _valid_sha256(digest):
                raise LineageError(f"{name} must be lowercase SHA-256")
        root = lineage_root_sha256
        if root is None:
            root = hashlib.sha256(_canonical_json({
                "policy_generation": POLICY_GENERATION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "state_schema_sha256": state_schema_sha256,
                "optimizer_identity_sha256": optimizer_identity_sha256,
                "atlas_catalog_sha256": atlas_catalog_sha256,
                "runtime_manifest_sha256": runtime_manifest_sha256,
                "training_config_sha256": training_config.sha256,
                "initialization": initialization,
            })).hexdigest()
        if not _valid_sha256(root):
            raise LineageError("lineage_root_sha256 must be lowercase SHA-256")
        return cls(
            checkpoint_format=CHECKPOINT_FORMAT,
            policy_generation=POLICY_GENERATION,
            feature_schema=FEATURE_SCHEMA,
            feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            observation_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            posture_classes=POSTURE_CLASSES,
            fire_classes=2,
            hook_classes=HOOK_CLASSES,
            weapon_classes=WEAPON_CLASSES,
            architecture=MODEL_ARCHITECTURE,
            state_schema_sha256=state_schema_sha256,
            optimizer_identity_sha256=optimizer_identity_sha256,
            atlas_catalog_sha256=atlas_catalog_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            training_config_json=training_config.to_json(),
            training_config_sha256=training_config.sha256,
            initialization=initialization,
            lineage_root_sha256=root,
            training_step=int(training_step),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MultiresCheckpointManifest":
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise LineageError(
                f"checkpoint manifest fields differ; missing={missing} extra={extra}"
            )
        integer_fields = {
            "observation_dim", "action_dim", "posture_classes", "fire_classes",
            "hook_classes", "weapon_classes", "training_step",
        }
        if any(type(value[name]) is not int for name in integer_fields):
            raise LineageError("checkpoint manifest has invalid integer field types")
        if any(
            not isinstance(value[name], str)
            for name in set(cls.__dataclass_fields__) - integer_fields
        ):
            raise LineageError("checkpoint manifest has invalid string field types")
        try:
            return cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except TypeError as error:
            raise LineageError("checkpoint manifest has invalid field types") from error

    def validate(
        self,
        *,
        expected_state_schema_sha256: str,
        expected_optimizer_identity_sha256: str,
        expected_atlas_catalog_sha256: str,
        expected_runtime_manifest_sha256: str,
        expected_training_config: MultiresTrainingConfiguration,
        expected_lineage_root_sha256: Optional[str] = None,
    ) -> None:
        expected = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "policy_generation": POLICY_GENERATION,
            "feature_schema": FEATURE_SCHEMA,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "observation_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "posture_classes": POSTURE_CLASSES,
            "fire_classes": 2,
            "hook_classes": HOOK_CLASSES,
            "weapon_classes": WEAPON_CLASSES,
            "architecture": MODEL_ARCHITECTURE,
            "state_schema_sha256": expected_state_schema_sha256,
            "optimizer_identity_sha256": expected_optimizer_identity_sha256,
            "atlas_catalog_sha256": expected_atlas_catalog_sha256,
            "runtime_manifest_sha256": expected_runtime_manifest_sha256,
            "training_config_json": expected_training_config.to_json(),
            "training_config_sha256": expected_training_config.sha256,
        }
        actual = asdict(self)
        mismatches = {
            name: (actual[name], wanted)
            for name, wanted in expected.items()
            if actual[name] != wanted
        }
        if self.initialization not in ALLOWED_INITIALIZATIONS:
            mismatches["initialization"] = (
                self.initialization, sorted(ALLOWED_INITIALIZATIONS)
            )
        if self.training_step < 0:
            mismatches["training_step"] = (self.training_step, ">= 0")
        if expected_lineage_root_sha256 is not None and (
            self.lineage_root_sha256 != expected_lineage_root_sha256
        ):
            mismatches["lineage_root_sha256"] = (
                self.lineage_root_sha256, expected_lineage_root_sha256
            )
        for name in (
            "state_schema_sha256",
            "optimizer_identity_sha256",
            "atlas_catalog_sha256",
            "runtime_manifest_sha256",
            "training_config_sha256",
            "lineage_root_sha256",
        ):
            if not _valid_sha256(str(actual[name])):
                mismatches[name] = (actual[name], "lowercase SHA-256")
        if mismatches:
            details = "; ".join(
                f"{name}={found!r} expected {wanted!r}"
                for name, (found, wanted) in sorted(mismatches.items())
            )
            raise LineageError(f"multires checkpoint attestation failed: {details}")
        try:
            parsed_training = MultiresTrainingConfiguration.from_json(
                self.training_config_json
            )
        except ValueError as error:
            raise LineageError("checkpoint training configuration is invalid") from error
        if parsed_training.sha256 != self.training_config_sha256:
            raise LineageError("checkpoint training configuration digest differs")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def policy_state_schema(policy: Any) -> tuple[str, tuple[dict[str, Any], ...]]:
    if getattr(policy, "policy_generation", None) != POLICY_GENERATION:
        raise LineageError("policy object is not the multires Atlas generation")
    if getattr(policy, "observation_dim", None) != OBS_DIM:
        raise LineageError("policy object does not declare the 298-input contract")
    state = policy.state_dict()
    return _tensor_schema(state)


def select_attested_checkpoint(path: Path) -> Path:
    """Select exactly one immutable path; never scan latest/legacy directories."""
    source = Path(path).expanduser()
    if source.is_symlink():
        raise LineageError("checkpoint selection rejects mutable symbolic links")
    if not source.is_file():
        raise LineageError(
            "checkpoint selection requires one explicit attested checkpoint file"
        )
    return source.resolve()


def save_attested_checkpoint(
    path: Path,
    policy: Any,
    *,
    atlas_catalog_sha256: str,
    runtime_manifest_sha256: str,
    training_config: MultiresTrainingConfiguration,
    initialization: str,
    training_step: int,
    optimizer: Any = None,
    lineage_root_sha256: Optional[str] = None,
) -> MultiresCheckpointManifest:
    """Atomically save one resumable same-generation checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise LineageError("PyTorch is required to save a policy checkpoint") from error
    state_schema_sha256, _rows = policy_state_schema(policy)
    optimizer_identity_sha256, _optimizer_descriptor = optimizer_identity(
        optimizer, policy
    )
    manifest = MultiresCheckpointManifest.create(
        state_schema_sha256=state_schema_sha256,
        optimizer_identity_sha256=optimizer_identity_sha256,
        atlas_catalog_sha256=atlas_catalog_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        training_config=training_config,
        initialization=initialization,
        training_step=training_step,
        lineage_root_sha256=lineage_root_sha256,
    )
    envelope = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "manifest": manifest.to_mapping(),
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(envelope, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def load_attested_checkpoint(
    path: Path,
    policy: Any,
    *,
    expected_atlas_catalog_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_training_config: MultiresTrainingConfiguration,
    optimizer: Any = None,
    expected_lineage_root_sha256: Optional[str] = None,
    map_location: Any = "cpu",
) -> MultiresCheckpointManifest:
    """Load only an exact attested envelope; there is no raw-state fallback."""
    try:
        import torch
    except ImportError as error:
        raise LineageError("PyTorch is required to load a policy checkpoint") from error
    source = select_attested_checkpoint(path)
    try:
        envelope = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as error:
        raise LineageError("checkpoint is not a safe weights-only envelope") from error
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "checkpoint_format", "manifest", "policy_state", "optimizer_state"
    }:
        raise LineageError(
            "legacy/raw checkpoint rejected: attested multires envelope required"
        )
    if envelope["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise LineageError("checkpoint envelope format is not multires v2")
    if not isinstance(envelope["manifest"], Mapping):
        raise LineageError("checkpoint manifest is missing")
    if not isinstance(envelope["policy_state"], Mapping):
        raise LineageError("checkpoint policy state is missing")
    if not isinstance(envelope["optimizer_state"], Mapping):
        raise LineageError("checkpoint optimizer state is missing")
    expected_schema, _rows = policy_state_schema(policy)
    expected_optimizer_identity, _optimizer_descriptor = optimizer_identity(
        optimizer, policy
    )
    loaded_optimizer_identity = _optimizer_state_identity(
        optimizer, envelope["optimizer_state"]
    )
    if loaded_optimizer_identity != expected_optimizer_identity:
        raise LineageError(
            "checkpoint optimizer state differs from the configured optimizer"
        )
    loaded_schema, _loaded_rows = _tensor_schema(envelope["policy_state"])
    if loaded_schema != expected_schema:
        raise LineageError("checkpoint tensor schema differs from fresh multires graph")
    manifest = MultiresCheckpointManifest.from_mapping(envelope["manifest"])
    manifest.validate(
        expected_state_schema_sha256=expected_schema,
        expected_optimizer_identity_sha256=expected_optimizer_identity,
        expected_atlas_catalog_sha256=expected_atlas_catalog_sha256,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        expected_training_config=expected_training_config,
        expected_lineage_root_sha256=expected_lineage_root_sha256,
    )
    # Mutation starts only after every manifest, tensor-shape, optimizer-class,
    # hyperparameter, and ownership check passed. Roll back both objects if a
    # framework-level loader still rejects the attested payload.
    original_policy_state = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
    try:
        policy.load_state_dict(envelope["policy_state"], strict=True)
        optimizer.load_state_dict(envelope["optimizer_state"])
    except Exception as error:
        policy.load_state_dict(original_policy_state, strict=True)
        optimizer.load_state_dict(original_optimizer_state)
        raise LineageError("checkpoint state transaction was rejected") from error
    return manifest
