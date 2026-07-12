"""Build and verify semantic runtime manifests for rollout workers.

The rollout wire format can prove that a worker used a particular policy
artifact, but a shape-valid batch is still unsafe if it came from a different
game binary, map, observation layout, reward configuration, or lattice build.
This module gives those inputs one canonical, path-independent identity.

Host-local paths and diagnostics are deliberately kept outside the signed
``semantic`` payload.  Two machines may install the same artifacts at
different paths and still attest to the same runtime.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


MANIFEST_SCHEMA = "q2-runtime-attestation-v1"
SIGNATURE_ALGORITHM = "hmac-sha256"

_MAP_SUFFIXES = (
    ".bsp",
    ".json",
    ".lattice.json",
    ".meta.json",
    ".chn",
    ".wnt",
    ".rtz",
)

# These values identify where/how a worker is launched, not the semantics of
# the trajectories it produces.  Artifact paths, worker IDs, transport
# credentials, port slabs, and output locations must not prevent two otherwise
# identical hosts from matching.
_NON_SEMANTIC_Q2_KEYS = {
    "Q2_BIND_IP",
    "Q2_CKPT_DIR",
    "Q2_HEAT_DIR",
    "Q2_ML_PORT_BASE",
    "Q2_RESUME_DIR",
    "Q2_ROOT",
    "Q2_RUN_TAG",
    "Q2_RUST_EXTENSION_PATH",
    "Q2_SERVER_STDOUT_LOG",
    "Q2_SOURCE_REVISION",
    "Q2_SV_PORT_BASE",
}
_NON_SEMANTIC_Q2_PREFIXES = (
    "Q2_DISTRIBUTED_",
    "Q2_OLLAMA_",
    "Q2_ROLLOUT_",
)
_ENV_LITERAL = re.compile(r"[\"']((?:Q2|R)_[A-Z0-9_]+)[\"']")


class AttestationError(ValueError):
    """The requested runtime cannot be represented or verified safely."""


def canonical_json(value: Any) -> bytes:
    """Return the one canonical encoding used for hashes and signatures."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, logical_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise AttestationError(f"required runtime artifact is missing: {path}")
    stat = path.stat()
    return {
        "name": logical_name,
        "sha256": sha256_file(path),
        "size": int(stat.st_size),
    }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_map_name(value: str) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".bsp") else name


def describe_maps(q2_root: Path, map_names: Iterable[str]) -> list[dict[str, Any]]:
    """Hash every runtime asset associated with each selected map.

    Both game and baseq2 search locations are represented because their
    relative location affects Quake's virtual filesystem precedence.
    """
    roots = (q2_root / "lithium" / "maps", q2_root / "baseq2" / "maps")
    result = []
    normalized_names = sorted({_normalize_map_name(name) for name in map_names})
    if not normalized_names:
        raise AttestationError("at least one map is required for runtime attestation")
    for name in normalized_names:
        files = []
        has_bsp = False
        expected_names = {name + suffix for suffix in _MAP_SUFFIXES}
        for directory in roots:
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir(), key=lambda item: item.name):
                if not path.is_file() or path.name not in expected_names:
                    continue
                relative = path.relative_to(q2_root).as_posix()
                files.append(_file_record(path, relative))
                has_bsp = has_bsp or path.name == f"{name}.bsp"
        if not has_bsp:
            raise AttestationError(f"map {name!r} has no BSP under {q2_root}")
        result.append({"name": name, "files": files})
    return result


def resolve_rust_extension(
    environment: Mapping[str, str],
    explicit_path: Optional[Path] = None,
) -> Optional[Path]:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    configured = environment.get("Q2_RUST_EXTENSION_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        import importlib.util

        spec = importlib.util.find_spec("q2_lattice_rs")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve()


def describe_observation(environment: Mapping[str, str]) -> dict[str, Any]:
    # Import constants only.  OBS_DIM itself is import-time environment
    # dependent, so total_dim is recomputed from the supplied environment.
    from harness import protocol

    ext_enabled = _truthy(environment.get("Q2_EXT_OBS", "0"))
    total = (
        int(protocol.OBS_BASE_DIM)
        + int(protocol.OBS_SESSION_MEMORY_DIM)
        + (int(protocol.OBS_EXT_DIM) if ext_enabled else 0)
    )
    return {
        "base_dim": int(protocol.OBS_BASE_DIM),
        "session_memory_dim": int(protocol.OBS_SESSION_MEMORY_DIM),
        "extension_dim": int(protocol.OBS_EXT_DIM),
        "extension_enabled": ext_enabled,
        "total_dim": total,
        "observation_packet_bytes": int(protocol.OBS_SIZE),
        "action_packet_bytes": int(protocol.ACT_SIZE),
        "observation_magic": int(protocol.ML_OBS_MAGIC),
        "action_magic": int(protocol.ML_ACT_MAGIC),
        "max_entities": int(protocol.ML_MAX_ENTITIES),
        "ray_count": int(protocol.ML_RAY_COUNT),
        "hook_zone_count": int(protocol.ML_HOOK_ZONES),
    }


def _shape_of(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise AttestationError("policy state contains a value without a tensor shape")
    return [int(dimension) for dimension in shape]


def _dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise AttestationError("policy state contains a value without a dtype")
    return str(dtype)


def describe_policy_state(
    state: Mapping[str, Any],
    *,
    observation_dim: int,
    action_dim: int = 8,
    hidden_dim: int = 256,
    architecture: str = "models.policy.Q2BotPolicy",
) -> dict[str, Any]:
    """Describe architecture from names/shapes/dtypes, never weight values."""
    tensors = []
    parameter_count = 0
    for name in sorted(state):
        shape = _shape_of(state[name])
        count = 1
        for dimension in shape:
            count *= dimension
        parameter_count += count
        tensors.append({"name": str(name), "shape": shape, "dtype": _dtype_of(state[name])})
    if not tensors:
        raise AttestationError("policy state is empty")
    schema_sha256 = sha256_bytes(canonical_json(tensors))
    return {
        "architecture": architecture,
        "observation_dim": int(observation_dim),
        "action_dim": int(action_dim),
        "hidden_dim": int(hidden_dim),
        "parameter_count": int(parameter_count),
        "state_schema_sha256": schema_sha256,
        "state_tensors": tensors,
    }


def load_policy_descriptor(
    observation_dim: int,
    checkpoint: Optional[Path] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Instantiate the policy (or load a checkpoint) and return semantics/diagnostics."""
    try:
        import torch
        from models.policy import ACTION_DIM, HIDDEN_DIM, OBS_DIM, Q2BotPolicy
    except ImportError as error:
        raise AttestationError(
            "PyTorch/model imports are required unless policy_descriptor is supplied"
        ) from error

    if int(OBS_DIM) != int(observation_dim):
        raise AttestationError(
            "model/manifest observation dimension mismatch: "
            f"{OBS_DIM} != {observation_dim}; set Q2_EXT_OBS before importing the policy"
        )
    policy = Q2BotPolicy()
    diagnostics: dict[str, Any] = {}
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise AttestationError(f"policy checkpoint is missing: {checkpoint}")
        loaded = torch.load(checkpoint, map_location="cpu")
        if isinstance(loaded, Mapping) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping):
            raise AttestationError("policy checkpoint does not contain a state mapping")
        policy.load_state_dict(loaded)
        diagnostics["checkpoint_path"] = str(checkpoint)
        diagnostics["checkpoint_sha256"] = sha256_file(checkpoint)
    descriptor = describe_policy_state(
        policy.state_dict(),
        observation_dim=observation_dim,
        action_dim=int(ACTION_DIM),
        hidden_dim=int(HIDDEN_DIM),
    )
    return descriptor, diagnostics


def _source_paths(source_root: Path) -> list[Path]:
    patterns = (
        "harness/**/*.py",
        "models/**/*.py",
        "train/**/*.py",
        "tools/rollout_worker.py",
        "tools/runtime_attestation.py",
        "tools/rollout_throughput_gate.py",
        "tools/rollout_throughput_probe.py",
        "Cargo.toml",
        "Cargo.lock",
        "crates/**/*.toml",
        "crates/**/*.rs",
    )
    paths = set()
    for pattern in patterns:
        paths.update(path for path in source_root.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(source_root).as_posix())


def _git_diagnostics(source_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).stdout.decode().strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).stdout
        return {
            "available": True,
            "head": revision,
            "dirty": bool(status),
            "status_sha256": sha256_bytes(status),
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False, "head": "", "dirty": None, "status_sha256": ""}


def describe_source(
    source_root: Path,
    source_revision: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_root.expanduser().resolve()
    paths = _source_paths(source_root)
    if not paths:
        raise AttestationError(f"no runtime source files found under {source_root}")
    files = [
        _file_record(path, path.relative_to(source_root).as_posix()) for path in paths
    ]
    git = _git_diagnostics(source_root)
    revision = str(source_revision or os.environ.get("Q2_SOURCE_REVISION", "")).strip()
    if not revision:
        revision = str(git.get("head", "")).strip()
    if not revision:
        raise AttestationError(
            "git revision is unavailable; pass --source-revision/Q2_SOURCE_REVISION"
        )
    semantic = {
        "git_revision": revision,
        "tree_sha256": sha256_bytes(canonical_json(files)),
        "files": files,
    }
    diagnostics = {"source_root": str(source_root), "git": git}
    return semantic, diagnostics


def _semantic_env_key(key: str) -> bool:
    if key in _NON_SEMANTIC_Q2_KEYS:
        return False
    if any(key.startswith(prefix) for prefix in _NON_SEMANTIC_Q2_PREFIXES):
        return False
    return key.startswith("R_") or key.startswith("Q2_")


def describe_environment(
    source_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Optional[str]]:
    """Capture every known reward/runtime override, including explicit unset.

    Source scanning makes an unset default visible in the manifest, while the
    source-tree fingerprint proves what that default means.
    """
    keys = {key for key in environment if _semantic_env_key(str(key))}
    for path in _source_paths(source_root):
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        keys.update(
            key for key in _ENV_LITERAL.findall(text) if _semantic_env_key(key)
        )
    keys.update(("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED"))
    return {
        key: (str(environment[key]) if key in environment else None)
        for key in sorted(keys)
    }


def _normalize_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise AttestationError(f"{label} is not canonical JSON data") from error


def semantic_digest(semantic: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json({"schema": MANIFEST_SCHEMA, "semantic": semantic}))


def _signature(digest: str, key: bytes) -> str:
    return hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()


def build_runtime_manifest(
    *,
    q2_root: Path,
    source_root: Path,
    map_names: Iterable[str],
    rust_extension: Optional[Path] = None,
    source_revision: Optional[str] = None,
    runtime_config: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, str]] = None,
    policy_descriptor: Optional[Mapping[str, Any]] = None,
    observation_descriptor: Optional[Mapping[str, Any]] = None,
    policy_checkpoint: Optional[Path] = None,
    hmac_key: Optional[bytes] = None,
) -> dict[str, Any]:
    """Build a sealed manifest from the worker's actual runtime inputs."""
    if hmac_key is not None and not hmac_key:
        raise AttestationError("manifest HMAC key must not be empty")
    q2_root = q2_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    env = dict(os.environ if environment is None else environment)
    observation = dict(
        observation_descriptor
        if observation_descriptor is not None
        else describe_observation(env)
    )
    policy_diagnostics: dict[str, Any] = {}
    if policy_descriptor is None:
        policy, policy_diagnostics = load_policy_descriptor(
            int(observation["total_dim"]), checkpoint=policy_checkpoint
        )
    else:
        policy = dict(policy_descriptor)
    if int(policy.get("observation_dim", -1)) != int(observation["total_dim"]):
        raise AttestationError(
            "policy/observation dimension mismatch: "
            f"{policy.get('observation_dim')} != {observation['total_dim']}"
        )

    rust_enabled = _truthy(env.get("Q2_RUST_LATTICE", "0"))
    rust_path = resolve_rust_extension(env, rust_extension)
    if rust_enabled and (rust_path is None or not rust_path.is_file()):
        raise AttestationError("Q2_RUST_LATTICE=1 but q2_lattice_rs is unavailable")
    rust_record: dict[str, Any] = {"enabled": rust_enabled}
    if rust_path is not None and rust_path.is_file():
        rust_record.update(_file_record(rust_path, "q2_lattice_rs"))

    source, source_diagnostics = describe_source(source_root, source_revision)
    semantic = {
        "artifacts": {
            "q2ded": _file_record(q2_root / "q2ded", "q2ded"),
            "game_module": _file_record(
                q2_root / "lithium" / "game.so", "lithium/game.so"
            ),
            "rust_lattice": rust_record,
        },
        "maps": describe_maps(q2_root, map_names),
        "observation": _normalize_json(observation, "observation descriptor"),
        "policy": _normalize_json(policy, "policy descriptor"),
        "environment": describe_environment(source_root, env),
        "runtime_config": _normalize_json(runtime_config or {}, "runtime config"),
        "source": source,
    }
    digest = semantic_digest(semantic)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "semantic": semantic,
        "manifest_sha256": digest,
        "diagnostics": {
            "q2_root": str(q2_root),
            "rust_extension_path": str(rust_path) if rust_path else "",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **source_diagnostics,
            **policy_diagnostics,
        },
    }
    if hmac_key is not None:
        manifest["signature"] = {
            "algorithm": SIGNATURE_ALGORITHM,
            "value": _signature(digest, hmac_key),
        }
    return manifest


def _diff_values(current: Any, expected: Any, path: str = "semantic") -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    # JSON's bool, integer, and floating-point values have distinct canonical
    # encodings even where Python equality aliases them (False == 0,
    # 1 == 1.0). A digest mismatch must never disappear from the structural
    # explanation or, worse, be accepted as semantically equal.
    if type(current) is not type(expected):
        return [{"path": path, "current": current, "expected": expected}]
    if isinstance(current, Mapping) and isinstance(expected, Mapping):
        for key in sorted(set(current) | set(expected)):
            child = f"{path}.{key}"
            if key not in current:
                differences.append({"path": child, "current": "<missing>", "expected": expected[key]})
            elif key not in expected:
                differences.append({"path": child, "current": current[key], "expected": "<missing>"})
            else:
                differences.extend(_diff_values(current[key], expected[key], child))
        return differences
    if isinstance(current, list) and isinstance(expected, list):
        if current != expected:
            differences.append({"path": path, "current": current, "expected": expected})
        return differences
    if current != expected:
        differences.append({"path": path, "current": current, "expected": expected})
    return differences


@dataclass(frozen=True)
class ManifestVerification:
    valid: bool
    digest: str
    errors: tuple[str, ...]
    differences: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "manifest_sha256": self.digest,
            "errors": list(self.errors),
            "differences": list(self.differences),
        }


def verify_runtime_manifest(
    manifest: Mapping[str, Any],
    *,
    expected: Optional[Mapping[str, Any]] = None,
    hmac_key: Optional[bytes] = None,
    require_signature: bool = False,
) -> ManifestVerification:
    errors = []
    differences: list[dict[str, Any]] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("unsupported manifest schema")
    semantic = manifest.get("semantic")
    if not isinstance(semantic, Mapping):
        errors.append("manifest semantic payload is missing")
        digest = ""
    else:
        digest = semantic_digest(semantic)
        if not hmac.compare_digest(digest, str(manifest.get("manifest_sha256", ""))):
            errors.append("manifest digest mismatch")

    signature = manifest.get("signature")
    if require_signature and not isinstance(signature, Mapping):
        errors.append("manifest signature is required")
    if require_signature and hmac_key is None:
        errors.append("manifest signature verification key is required")
    if hmac_key is not None:
        if not hmac_key:
            errors.append("manifest HMAC key must not be empty")
        if not isinstance(signature, Mapping):
            errors.append("manifest signature is missing")
        elif signature.get("algorithm") != SIGNATURE_ALGORITHM:
            errors.append("unsupported manifest signature algorithm")
        elif not hmac.compare_digest(
            str(signature.get("value", "")), _signature(digest, hmac_key)
        ):
            errors.append("manifest signature mismatch")

    if expected is not None:
        expected_result = verify_runtime_manifest(
            expected,
            hmac_key=hmac_key,
            require_signature=require_signature,
        )
        if not expected_result.valid:
            errors.extend(f"expected: {error}" for error in expected_result.errors)
        expected_semantic = expected.get("semantic")
        if isinstance(semantic, Mapping) and isinstance(expected_semantic, Mapping):
            differences.extend(_diff_values(semantic, expected_semantic))
            if differences:
                errors.append("semantic manifest does not match expected runtime")

    return ManifestVerification(
        valid=not errors,
        digest=digest,
        errors=tuple(errors),
        differences=tuple(differences),
    )


def load_runtime_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttestationError(f"could not read runtime manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise AttestationError("runtime manifest must be a JSON object")
    return value


def write_runtime_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)
