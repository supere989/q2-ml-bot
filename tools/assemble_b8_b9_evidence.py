#!/usr/bin/env python3
"""Assemble fail-closed B8 quality/shadow and B9 promotion-decision evidence.

This tool is deliberately a consumer, never a workload launcher or deployment
selector.  ``season`` evaluates one read-only generated or stock archive,
``b8`` joins both independent season gates with a public-topology shadow, and
``b9`` joins the green B8 gate with a cold-restart archive.  Even a green B9
decision records that no public mutation occurred and requires a separate root
promotion action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_catalog import ATLAS_CATALOG_DOMAIN, ATLAS_CATALOG_SCHEMA
from harness.runtime_attestation import verify_runtime_manifest
from tools.assemble_b6_wsl_g1_campaign import validate_campaign as validate_b6_campaign
from tools.verify_multires_integration import (
    GATE_ORDER as INTEGRATION_GATE_ORDER,
    canonical_report_bytes as integration_report_bytes,
    run_gates as rerun_integration_gates,
)


ARCHIVE_SCHEMA = "q2-multires-quality-archive-v1"
SEASON_EVALUATION_SCHEMA = "q2-multires-b8-season-evaluation-v1"
SHADOW_EVIDENCE_SCHEMA = "q2-multires-b8-shadow-evidence-v1"
COLD_RESTART_SCHEMA = "q2-multires-b9-cold-restart-evidence-v1"
RECONSTRUCTION_MANIFEST_SCHEMA = "q2-multires-b9-reconstruction-manifest-v1"
SEASON_GATE_SCHEMA = "q2-multires-b8-season-gate-v1"
B8_GATE_SCHEMA = "q2-multires-b8-gate-v1"
B9_DECISION_SCHEMA = "q2-multires-b9-promotion-decision-v1"
CURRENT_SEASON_SCHEMA = "q2-multires-current-season-v1"
B7_GATE_SCHEMA = "q2-multires-b7-predecessor-gate-v1"
LEGACY_ABSENCE_SCHEMA = "q2-multires-legacy-selector-absence-v1"
TOOL = "assemble_b8_b9_evidence"
G0_SUMMARY_SCHEMA = "q2-multires-b8-g0-evidence-summary-v1"

OUTPUT_SCHEMA_FILES = {
    ARCHIVE_SCHEMA: "q2-multires-quality-archive-v1.schema.json",
    SEASON_GATE_SCHEMA: "q2-multires-b8-season-gate-v1.schema.json",
    B8_GATE_SCHEMA: "q2-multires-b8-gate-v1.schema.json",
    RECONSTRUCTION_MANIFEST_SCHEMA:
        "q2-multires-b9-reconstruction-manifest-v1.schema.json",
    B9_DECISION_SCHEMA: "q2-multires-b9-promotion-decision-v1.schema.json",
}

MIN_TRANSITIONS = 16_384
MIN_ACTION_ECHO_ACCEPTANCE = 0.97
MIN_VERTICAL_ECHO_MATCH = 0.99
MAX_DOWNLOOK_RATE = 0.10
MAX_MOVING_PITCH_DEG = 10.0
MAX_BACKWARD_RATE = 0.40
MAX_GUIDE_DEGRADATION = 0.15
MIN_ALIGNED_FIRE_PRECISION = 0.85
MAX_YAW_MAE_DEG = 12.0
MAX_PITCH_MAE_DEG = 8.0
CLIENTS = 4
MAXCLIENTS = 6
HUMAN_SLOTS = 2

_SHA = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")
_GIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

INTEGRATION_EVIDENCE_ROLES = {
    name: f"integration_{name}" for name in INTEGRATION_GATE_ORDER
}
INTEGRATION_ROLES = frozenset({
    "integration_envelope", "integration_report",
    *INTEGRATION_EVIDENCE_ROLES.values(),
})
SEASON_ROLES = frozenset({
    "current_season", "evaluation", "b7_gate", "b7_stage_evaluation",
    "runtime_manifest", "policy", "atlas", "atlas_catalog", "dyn_schema",
    "reward_configuration", "source_identity", "legacy_selector_absence",
    "retirement_manifest", *INTEGRATION_ROLES,
})
SHADOW_ROLES = frozenset({
    "shadow_evidence", "runtime_manifest", "policy",
    "atlas", "atlas_catalog", "dyn_schema", "reward_configuration", "source_identity",
    "legacy_selector_absence", "retirement_manifest", *INTEGRATION_ROLES,
})
RESTART_ROLES = frozenset({
    "cold_restart_evidence", "reconstruction_manifest", "b8_gate",
    "runtime_manifest", "policy", "atlas", "atlas_catalog", "dyn_schema",
    "reward_configuration", "source_identity", "legacy_selector_absence",
    "retirement_manifest", *INTEGRATION_ROLES,
})
IDENTITY_ROLES = (
    "integration_envelope", "integration_report", "runtime_manifest", "policy", "atlas",
    "atlas_catalog", "dyn_schema", "reward_configuration", "source_identity",
    "legacy_selector_absence", "retirement_manifest",
    *INTEGRATION_EVIDENCE_ROLES.values(),
)
STABLE_IDENTITY_ROLES = tuple(
    role for role in IDENTITY_ROLES if role != "atlas"
)
SEASON_DISTINCT_FIELDS = (
    "season_id",
    "archive_id",
    "archive_manifest_sha256",
    "archive_manifest_file_sha256",
    "current_season_file_sha256",
    "evaluation_file_sha256",
    "current_season_evidence_sha256",
    "evaluation_evidence_sha256",
    "causal_metrics_window_sha256",
    "network_metrics_window_sha256",
    "atlas_sha256",
    "atlas_artifact_sha256",
)


class GateError(RuntimeError):
    """The supplied bytes cannot support a green B8/B9 conclusion."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise GateError(f"non-finite JSON constant {value!r}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GateError("value is not canonical JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    source = Path(os.path.abspath(path.expanduser()))
    _require(source.is_absolute(), f"{label} must be absolute")
    _require(not source.is_symlink() and source.is_file(),
             f"{label} must be a regular non-symlink file")
    return source


def load_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular(path, label)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not strict UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(actual == expected, f"{label} fields differ; "
             f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             f"{label} must be a non-placeholder lowercase SHA-256")
    return str(value)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()),
             f"{label} must be a nonempty string")
    lowered = value.lower()
    _require(not any(word in lowered for word in ("placeholder", "todo", "tbd")),
             f"{label} contains a placeholder")
    return str(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum,
             f"{label} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, label: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} must be finite numeric evidence")
    result = float(value)
    if minimum is not None:
        _require(result >= minimum, f"{label} must be >= {minimum}")
    if maximum is not None:
        _require(result <= maximum, f"{label} must be <= {maximum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be boolean")
    return bool(value)


def _schema_target(root: Mapping[str, Any], reference: str,
                   label: str) -> Mapping[str, Any]:
    _require(reference.startswith("#/"), f"{label} has a non-local schema reference")
    value: Any = root
    for raw in reference[2:].split("/"):
        component = raw.replace("~1", "/").replace("~0", "~")
        _require(isinstance(value, Mapping) and component in value,
                 f"{label} has an unresolved schema reference {reference!r}")
        value = value[component]
    return _mapping(value, f"{label} schema reference {reference}")


def _audit_checked_schema(node: Mapping[str, Any], label: str) -> None:
    supported = {
        "$schema", "$id", "$defs", "$ref", "title", "type", "const", "enum",
        "required", "properties", "additionalProperties", "minProperties",
        "maxProperties", "items", "minimum", "maximum", "minLength", "pattern",
        "allOf",
    }
    unknown = set(node) - supported
    _require(not unknown,
             f"{label} uses unsupported schema keywords {sorted(unknown)}")
    for container_name in ("$defs", "properties"):
        container = node.get(container_name, {})
        _require(isinstance(container, Mapping),
                 f"{label} {container_name} is malformed")
        for name, child in container.items():
            _audit_checked_schema(
                _mapping(child, f"{label} {container_name}.{name}"),
                f"{label} {container_name}.{name}",
            )
    for child_name in ("additionalProperties", "items"):
        child = node.get(child_name)
        if isinstance(child, Mapping):
            _audit_checked_schema(child, f"{label} {child_name}")
    all_of = node.get("allOf", [])
    _require(isinstance(all_of, list), f"{label} allOf is malformed")
    for index, child in enumerate(all_of):
        _audit_checked_schema(
            _mapping(child, f"{label} allOf[{index}]"),
            f"{label} allOf[{index}]",
        )


def _checked_schema_node(value: Any, node: Mapping[str, Any],
                         root: Mapping[str, Any], label: str) -> None:
    """Evaluate the checked-in schema subset used by B8/B9 without an optional dep."""
    if "$ref" in node:
        _checked_schema_node(
            value, _schema_target(root, str(node["$ref"]), label), root, label,
        )
    for index, child in enumerate(node.get("allOf", ())):
        _checked_schema_node(
            value, _mapping(child, f"{label} allOf[{index}]"), root, label,
        )
    if "const" in node:
        wanted = node["const"]
        equal = value == wanted and not (
            isinstance(value, bool) != isinstance(wanted, bool)
        )
        _require(equal, f"{label} differs from schema const")
    if "enum" in node:
        _require(any(
            value == wanted and not (
                isinstance(value, bool) != isinstance(wanted, bool)
            ) for wanted in node["enum"]
        ), f"{label} is outside the schema enum")

    expected_type = node.get("type")
    if expected_type is not None:
        names = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        matches = {
            "object": lambda item: isinstance(item, Mapping),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: type(item) is int,
            "number": lambda item: type(item) in (int, float) and
                math.isfinite(float(item)),
            "boolean": lambda item: type(item) is bool,
            "null": lambda item: item is None,
        }
        _require(all(name in matches for name in names),
                 f"{label} schema uses an unsupported type")
        _require(any(matches[name](value) for name in names),
                 f"{label} differs from schema type {names}")

    if isinstance(value, str):
        if "minLength" in node:
            _require(len(value) >= int(node["minLength"]),
                     f"{label} is shorter than the schema minimum")
        if "pattern" in node:
            _require(re.fullmatch(str(node["pattern"]), value) is not None,
                     f"{label} differs from the schema pattern")
    if type(value) in (int, float):
        if "minimum" in node:
            _require(value >= node["minimum"],
                     f"{label} is below the schema minimum")
        if "maximum" in node:
            _require(value <= node["maximum"],
                     f"{label} is above the schema maximum")
    if isinstance(value, list) and "items" in node:
        item_schema = _mapping(node["items"], f"{label} item schema")
        for index, item in enumerate(value):
            _checked_schema_node(item, item_schema, root, f"{label}[{index}]")
    if isinstance(value, Mapping):
        required = node.get("required", [])
        _require(isinstance(required, list), f"{label} schema required is malformed")
        missing = set(required) - set(value)
        _require(not missing, f"{label} schema fields differ; missing={sorted(missing)}")
        if "minProperties" in node:
            _require(len(value) >= int(node["minProperties"]),
                     f"{label} has fewer fields than the schema minimum")
        if "maxProperties" in node:
            _require(len(value) <= int(node["maxProperties"]),
                     f"{label} has more fields than the schema maximum")
        properties = _mapping(node.get("properties", {}),
                              f"{label} schema properties")
        extras = set(value) - set(properties)
        additional = node.get("additionalProperties", True)
        if additional is False:
            _require(not extras,
                     f"{label} schema fields differ; extra={sorted(extras)}")
        elif isinstance(additional, Mapping):
            for name in sorted(extras):
                _checked_schema_node(
                    value[name], additional, root, f"{label}.{name}",
                )
        for name in sorted(set(value) & set(properties)):
            _checked_schema_node(
                value[name], _mapping(properties[name], f"{label}.{name} schema"),
                root, f"{label}.{name}",
            )


def _validate_checked_schema(value: Mapping[str, Any], schema: str,
                             label: str) -> None:
    filename = OUTPUT_SCHEMA_FILES.get(schema)
    _require(filename is not None, f"{label} has no checked-in schema mapping")
    schema_path = ROOT / "schemas" / filename
    schema_value = load_json(schema_path, f"{label} checked-in schema")
    _audit_checked_schema(schema_value, f"{label} checked-in schema")
    _checked_schema_node(value, schema_value, schema_value, label)


def _validate_seal(value: Mapping[str, Any], schema: str, label: str,
                   field: str = "evidence_sha256") -> str:
    _require(value.get("schema") == schema, f"{label} schema differs")
    body = dict(value)
    digest = _digest(body.pop(field, None), f"{label} {field}")
    _require(digest == canonical_sha256(body), f"{label} seal differs")
    return digest


def _record(path: Path) -> dict[str, Any]:
    source = _regular(path, str(path))
    return {"bytes": source.stat().st_size, "sha256": _file_sha256(source)}


def _identity_records(files: Mapping[str, Path]) -> dict[str, Any]:
    return {role: _record(files[role]) for role in IDENTITY_ROLES}


def _atlas_catalog_identity(
    path: Path, *, expected_sha256: str | None = None,
    active_atlas_sha256: str | None = None,
) -> str:
    catalog = load_json(path, "Atlas catalog")
    _exact_keys(catalog, {
        "schema", "domain", "map_count", "maps", "atlas_catalog_sha256",
    }, "Atlas catalog")
    _require(catalog["schema"] == ATLAS_CATALOG_SCHEMA,
             "Atlas catalog schema differs")
    _require(catalog["domain"] == ATLAS_CATALOG_DOMAIN,
             "Atlas catalog domain differs")
    maps = catalog["maps"]
    _require(isinstance(maps, list) and maps and catalog["map_count"] == len(maps),
             "Atlas catalog map inventory differs")
    semantic_sha256 = canonical_sha256({
        "domain": catalog["domain"], "maps": maps,
    })
    _require(catalog["atlas_catalog_sha256"] == semantic_sha256,
             "Atlas catalog semantic seal differs")
    if expected_sha256 is not None:
        _require(semantic_sha256 == expected_sha256,
                 "Atlas catalog differs from current season identity")
    if active_atlas_sha256 is not None:
        _require(any(
            isinstance(record, Mapping)
            and record.get("atlas_sha256") == active_atlas_sha256
            for record in maps
        ), "active Atlas is absent from the admitted catalog")
    return semantic_sha256


def _validate_identity_artifacts(files: Mapping[str, Path],
                                 current: Mapping[str, Any] | None = None) -> None:
    _require("atlas_catalog" in files,
             "identity artifacts lack the admitted Atlas catalog")
    _atlas_catalog_identity(
        files["atlas_catalog"],
        expected_sha256=None if current is None
        else str(current.get("atlas_catalog_sha256")),
        active_atlas_sha256=None if current is None
        else str(current.get("atlas_sha256")),
    )
    envelope = load_json(files["integration_envelope"], "integration envelope")
    _require(envelope.get("schema") == "multires-integration-evidence-v1",
             "integration envelope schema differs")
    evidence = _mapping(envelope.get("evidence"), "integration evidence map")
    _require(set(evidence) == set(INTEGRATION_GATE_ORDER),
             "integration envelope gate inventory differs")
    base = files["integration_envelope"].parent.resolve()
    for gate_name in INTEGRATION_GATE_ORDER:
        relative = _safe_relative(evidence[gate_name],
                                  f"integration {gate_name} path")
        resolved = base.joinpath(*relative.parts).resolve()
        expected = files[INTEGRATION_EVIDENCE_ROLES[gate_name]].resolve()
        _require(resolved == expected and resolved.is_relative_to(base),
                 f"integration {gate_name} path does not bind the archived evidence")

    integration = load_json(files["integration_report"], "integration report")
    recomputed = rerun_integration_gates(files["integration_envelope"])
    _require(recomputed.get("overall") == "pass" and
             recomputed.get("failed_gates") == [],
             "archived integration evidence rerun is red: " +
             ",".join(recomputed.get("failed_gates", [])))
    _require(integration == recomputed, "archived integration report is stale")
    expected_report_bytes = integration_report_bytes(recomputed) + b"\n"
    _require(files["integration_report"].read_bytes() == expected_report_bytes,
             "archived integration report bytes are not the canonical rerun")

    b6 = load_json(files[INTEGRATION_EVIDENCE_ROLES["wsl_b6_campaign"]],
                   "archived B6 campaign")
    try:
        validate_b6_campaign(b6)
    except Exception as error:
        raise GateError(f"archived B6 campaign is invalid: {error}") from error
    b6_sources = _mapping(b6.get("source_repositories"), "B6 source repositories")

    runtime = load_json(files["runtime_manifest"], "runtime manifest")
    verification = verify_runtime_manifest(runtime)
    _require(verification.valid, "runtime manifest is invalid: " +
             "; ".join(verification.errors))
    b6_bindings = _mapping(b6.get("bindings"), "B6 bindings")
    _require(b6_bindings.get("runtime_manifest_identity_sha256") ==
             verification.digest,
             "runtime manifest semantic identity differs from archived B6")

    source = load_json(files["source_identity"], "source identity")
    _validate_seal(source, "q2-multires-source-identity-v1", "source identity")
    _exact_keys(source, {"schema", "repositories", "clean", "evidence_sha256"},
                "source identity")
    repositories = _mapping(source["repositories"], "source repositories")
    _require(set(repositories) == {"bot", "client", "game"},
             "source identity repository set differs")
    _require(source["clean"] is True, "source identity is not clean")
    for name, record in repositories.items():
        identity = _mapping(record, f"source {name}")
        _exact_keys(identity, {"commit", "tree", "clean"}, f"source {name}")
        _require(identity["clean"] is True, f"source {name} is not clean")
        for field in ("commit", "tree"):
            _require(isinstance(identity[field], str) and
                     _GIT.fullmatch(identity[field]) is not None,
                     f"source {name} {field} is malformed")
    _require(dict(repositories) == dict(b6_sources),
             "source identity differs from archived B6 source repositories")

    retirement = load_json(files["retirement_manifest"], "retirement manifest")
    _require(retirement.get("schema") == "q2-multires-runtime-retirement-v1" and
             retirement.get("status") == "legacy-runtime-retired" and
             retirement.get("fallback_allowed") is False,
             "retirement manifest does not enforce one-way retirement")
    runtime_config = _mapping(_mapping(runtime.get("semantic"),
                                       "runtime semantic").get("runtime_config"),
                              "runtime configuration")
    _require(runtime_config.get("retirement_manifest_sha256") ==
             _file_sha256(files["retirement_manifest"]),
             "runtime semantic retirement binding differs")
    b6_retirement = _mapping(
        b6_bindings.get("retirement_manifest"), "B6 retirement binding",
    )
    _require(b6_retirement.get("sha256") ==
             _file_sha256(files["retirement_manifest"]),
             "retirement manifest bytes differ from archived B6")

    if current is not None:
        _require(verification.digest == current["runtime_manifest_sha256"],
                 "runtime manifest semantic identity differs from current season")
        _require(_file_sha256(files["atlas"]) == current["atlas_sha256"],
                 "Atlas bytes differ from current season identity")
        _require(_file_sha256(files["policy"]) == current.get("checkpoint_sha256"),
                 "policy bytes differ from current season checkpoint identity")


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any], label: str,
                   roles: Sequence[str] = STABLE_IDENTITY_ROLES) -> None:
    for role in roles:
        _require(left.get(role) == right.get(role),
                 f"{label} {role} byte identity differs")


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    text = _text(value, label)
    path = PurePosixPath(text)
    _require(not path.is_absolute() and path.parts and all(
        part not in ("", ".", "..") for part in path.parts
    ), f"{label} must be a normalized relative path")
    _require(str(path) == text, f"{label} is not normalized POSIX syntax")
    return path


def _is_read_only(path: Path) -> bool:
    return not bool(stat.S_IMODE(path.stat().st_mode) & 0o222)


def validate_archive(manifest_path: Path, *, kind: str) -> tuple[
    dict[str, Any], dict[str, Path]
]:
    manifest_source = _regular(manifest_path, "archive manifest")
    _require(manifest_source.stat().st_nlink == 1,
             "archive manifest must not be hard-linked")
    manifest = load_json(manifest_source, "archive manifest")
    _validate_seal(manifest, ARCHIVE_SCHEMA, "archive manifest", "manifest_sha256")
    _validate_checked_schema(manifest, ARCHIVE_SCHEMA, "archive manifest")
    expected_roles = {
        "season": SEASON_ROLES, "shadow": SHADOW_ROLES, "restart": RESTART_ROLES,
    }.get(kind)
    _require(expected_roles is not None, f"unsupported archive kind {kind!r}")
    _exact_keys(manifest, {
        "schema", "archive_id", "archive_kind", "season_kind", "read_only",
        "files", "manifest_sha256",
    }, "archive manifest")
    _text(manifest["archive_id"], "archive id")
    _require(manifest["archive_kind"] == kind, "archive kind differs")
    if kind == "season":
        _require(manifest["season_kind"] in ("generated", "stock"),
                 "season archive kind must be generated or stock")
    else:
        _require(manifest["season_kind"] is None,
                 "non-season archive cannot claim a season kind")
    _require(manifest["read_only"] is True, "archive does not declare read-only")
    entries = _mapping(manifest["files"], "archive files")
    _require(set(entries) == set(expected_roles), "archive roles differ; "
             f"missing={sorted(set(expected_roles)-set(entries))} "
             f"extra={sorted(set(entries)-set(expected_roles))}")

    root = manifest_source.parent
    _require(not root.is_symlink() and root.is_dir(), "archive root is invalid")
    _require(_is_read_only(root), "archive root remains writable")
    _require(_is_read_only(manifest_source), "archive manifest remains writable")
    files: dict[str, Path] = {}
    declared: set[str] = {manifest_source.name}
    for role in sorted(expected_roles):
        entry = _mapping(entries[role], f"archive role {role}")
        _exact_keys(entry, {"path", "bytes", "sha256"}, f"archive role {role}")
        relative = _safe_relative(entry["path"], f"archive role {role} path")
        source = root.joinpath(*relative.parts)
        _regular(source, f"archive role {role}")
        _require(_is_read_only(source), f"archive role {role} remains writable")
        _require(source.resolve().is_relative_to(root.resolve()),
                 f"archive role {role} escapes the archive root")
        _require(source.stat().st_size == _integer(
            entry["bytes"], f"archive role {role} bytes", minimum=1,
        ), f"archive role {role} byte count differs")
        _require(_file_sha256(source) == _digest(
            entry["sha256"], f"archive role {role} SHA-256",
        ), f"archive role {role} SHA-256 differs")
        _require(str(relative) not in declared,
                 f"archive roles alias {relative}")
        declared.add(str(relative))
        files[role] = source

    discovered: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        _require(not directory_path.is_symlink() and _is_read_only(directory_path),
                 f"archive directory {directory_path} is symlinked or writable")
        for name in names:
            child = directory_path / name
            _require(not child.is_symlink(), f"archive directory symlink rejected: {child}")
        for name in filenames:
            child = directory_path / name
            _require(not child.is_symlink() and child.is_file(),
                     f"archive non-regular member rejected: {child}")
            _require(child.stat().st_nlink == 1,
                     f"archive hard-linked member rejected: {child}")
            discovered.add(child.relative_to(root).as_posix())
    _require(discovered == declared, "archive membership differs; "
             f"missing={sorted(declared-discovered)} extra={sorted(discovered-declared)}")
    return manifest, files


def _artifact_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        _require(bool(separator) and bool(role) and bool(raw_path),
                 "artifact must use ROLE=/absolute/path syntax")
        _require(role not in result, f"duplicate archive artifact role {role!r}")
        result[role] = _regular(Path(raw_path), f"archive source {role}")
    return result


def author_archive(*, destination: Path, kind: str, archive_id: str,
                   season_kind: str | None, artifacts: Mapping[str, Path]) -> Path:
    """Copy an exact evidence set and atomically publish one read-only archive."""
    expected_roles = {
        "season": SEASON_ROLES, "shadow": SHADOW_ROLES, "restart": RESTART_ROLES,
    }.get(kind)
    _require(expected_roles is not None, f"unsupported archive kind {kind!r}")
    required_sources = set(expected_roles) - set(INTEGRATION_EVIDENCE_ROLES.values())
    _require(set(artifacts) == required_sources, "archive source roles differ; "
             f"missing={sorted(required_sources-set(artifacts))} "
             f"extra={sorted(set(artifacts)-required_sources)}")
    _text(archive_id, "archive id")
    if kind == "season":
        _require(season_kind in ("generated", "stock"),
                 "season archive requires --season-kind generated|stock")
    else:
        _require(season_kind is None, "non-season archive rejects --season-kind")
    output = Path(os.path.abspath(destination.expanduser()))
    _require(not output.exists() and not output.is_symlink(),
             "exclusive archive output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.parent.is_symlink() and output.parent.is_dir(),
             "archive output parent is invalid")
    source_paths = [path.resolve() for path in artifacts.values()]
    _require(len(set(source_paths)) == len(source_paths),
             "archive source roles alias the same file")
    _require(all(not path.is_relative_to(output) for path in source_paths),
             "archive source cannot be inside its output")

    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.incoming-", dir=output.parent,
    ))
    copied: dict[str, Path] = {}
    try:
        artifact_root = staging / "artifacts"
        artifact_root.mkdir()
        for role, source in sorted(artifacts.items()):
            if role == "integration_envelope":
                continue
            target = artifact_root / role
            shutil.copyfile(source, target, follow_symlinks=False)
            copied[role] = target

        source_envelope = load_json(
            artifacts["integration_envelope"], "source integration envelope",
        )
        _require(source_envelope.get("schema") == "multires-integration-evidence-v1",
                 "source integration envelope schema differs")
        source_map = _mapping(source_envelope.get("evidence"),
                              "source integration evidence map")
        _require(set(source_map) == set(INTEGRATION_GATE_ORDER),
                 "source integration evidence inventory differs")
        integration_root = staging / "integration"
        integration_root.mkdir()
        envelope_target = integration_root / "envelope.json"
        shutil.copyfile(artifacts["integration_envelope"], envelope_target,
                        follow_symlinks=False)
        copied["integration_envelope"] = envelope_target
        source_base = artifacts["integration_envelope"].parent.resolve()
        used_targets = {envelope_target.resolve()}
        for gate_name in INTEGRATION_GATE_ORDER:
            relative = _safe_relative(source_map[gate_name],
                                      f"source integration {gate_name} path")
            source = source_base.joinpath(*relative.parts)
            _regular(source, f"source integration {gate_name} evidence")
            _require(source.resolve().is_relative_to(source_base),
                     f"source integration {gate_name} escapes its bundle")
            target = integration_root.joinpath(*relative.parts)
            _require(target.resolve() not in used_targets,
                     f"integration evidence target aliases {target}")
            used_targets.add(target.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
            copied[INTEGRATION_EVIDENCE_ROLES[gate_name]] = target

        _require(set(copied) == set(expected_roles),
                 "archive author did not materialize the exact role inventory")
        current = (
            load_json(copied["current_season"], "archive source current season")
            if kind == "season" else None
        )
        if current is not None:
            _current_season(current)
        _validate_identity_artifacts(copied, current)
        body = {
            "schema": ARCHIVE_SCHEMA,
            "archive_id": archive_id,
            "archive_kind": kind,
            "season_kind": season_kind,
            "read_only": True,
            "files": {
                role: {
                    "path": path.relative_to(staging).as_posix(), **_record(path),
                }
                for role, path in sorted(copied.items())
            },
        }
        manifest = staging / "archive-manifest.json"
        manifest_value = {
            **body, "manifest_sha256": canonical_sha256(body),
        }
        manifest.write_bytes(
            (json.dumps(manifest_value, sort_keys=True, indent=2,
                        allow_nan=False) + "\n").encode("utf-8")
        )
        for directory, _, filenames in os.walk(staging):
            for filename in filenames:
                path = Path(directory) / filename
                path.chmod(0o444)
        for directory, _, _ in list(os.walk(staging, topdown=False)):
            Path(directory).chmod(0o555)
        os.rename(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return output / "archive-manifest.json"
    except Exception:
        if staging.exists():
            for directory, names, filenames in os.walk(staging, topdown=False):
                for filename in filenames:
                    (Path(directory) / filename).chmod(0o600)
                for name in names:
                    (Path(directory) / name).chmod(0o700)
                Path(directory).chmod(0o700)
            shutil.rmtree(staging)
        raise


def _legacy_absent(value: Mapping[str, Any]) -> str:
    digest = _validate_seal(
        value, LEGACY_ABSENCE_SCHEMA, "legacy selector absence report",
    )
    _exact_keys(value, {
        "schema", "legacy_runtime_selectors", "legacy_policy_selectors",
        "legacy_optimizer_selectors", "legacy_dyn_selectors",
        "legacy_protocol_selectors", "operational_fallbacks", "passed",
        "evidence_sha256",
    }, "legacy selector absence report")
    for name in (
        "legacy_runtime_selectors", "legacy_policy_selectors",
        "legacy_optimizer_selectors", "legacy_dyn_selectors",
        "legacy_protocol_selectors", "operational_fallbacks",
    ):
        _require(value[name] == [], f"legacy selector absence {name} is not empty")
    _require(value["passed"] is True, "legacy selector absence report is red")
    return digest


def _validate_b7_gate(value: Mapping[str, Any], files: Mapping[str, Path]) -> str:
    digest = _validate_seal(value, B7_GATE_SCHEMA, "B7 stage-7 gate")
    expected = {
        "decision": "passed", "stage_id": 7,
        "stage_name": "full-guide-off-ablation", "automatic_promotion": False,
    }
    for name, wanted in expected.items():
        _require(value.get(name) == wanted, f"B7 gate {name} differs")
    _digest(value.get("stage_configuration_sha256"), "B7 stage configuration")
    _digest(value.get("runtime_manifest_sha256"), "B7 runtime manifest")
    _digest(value.get("atlas_catalog_sha256"), "B7 Atlas catalog")
    _digest(value.get("lineage_root_sha256"), "B7 lineage root")
    _integer(value.get("accepted_transitions"), "B7 accepted transitions", minimum=1)
    _integer(value.get("policy_updates"), "B7 policy updates", minimum=1)
    _integer(value.get("optimizer_steps"), "B7 optimizer steps", minimum=1)
    artifacts = _mapping(value.get("artifacts"), "B7 gate artifacts")
    _require(set(artifacts) == {"completed_stage", "evaluator"},
             "B7 gate lacks exact completion/evaluator bindings")
    for label, record in artifacts.items():
        binding = _mapping(record, f"B7 {label} binding")
        _exact_keys(binding, {"path", "sha256"}, f"B7 {label} binding")
        _text(binding["path"], f"B7 {label} path")
        _digest(binding["sha256"], f"B7 {label} SHA-256")
    _require(artifacts["completed_stage"]["sha256"] ==
             _file_sha256(files["current_season"]),
             "B7 completed-stage binding differs from archived current season")
    _require(artifacts["evaluator"]["sha256"] ==
             _file_sha256(files["b7_stage_evaluation"]),
             "B7 evaluator binding differs from archived stage evaluator")
    stage_evaluator = load_json(
        files["b7_stage_evaluation"], "B7 stage-7 evaluator",
    )
    _validate_seal(
        stage_evaluator, "q2-multires-b7-stage-evaluation-v1",
        "B7 stage-7 evaluator",
    )
    _exact_keys(stage_evaluator, {
        "schema", "decision", "stage_id", "stage_name",
        "stage_configuration_sha256", "runtime_manifest_sha256",
        "atlas_sha256", "atlas_catalog_sha256", "lineage_root_sha256", "completed_season",
        "minimum_accepted_transitions", "predicates", "stage_specific",
        "automatic_promotion", "evidence_sha256",
    }, "B7 stage-7 evaluator")
    _require(files["b7_stage_evaluation"].read_bytes() ==
             canonical_bytes(stage_evaluator) + b"\n",
             "B7 stage evaluator bytes are not canonical")
    for name, wanted in (
        ("decision", "passed"), ("stage_id", 7),
        ("stage_name", "full-guide-off-ablation"),
        ("runtime_manifest_sha256", value["runtime_manifest_sha256"]),
        ("lineage_root_sha256", value["lineage_root_sha256"]),
        ("automatic_promotion", False),
    ):
        _require(stage_evaluator.get(name) == wanted,
                 f"B7 stage evaluator {name} differs")
    _require(stage_evaluator["stage_configuration_sha256"] ==
             value["stage_configuration_sha256"],
             "B7 stage evaluator configuration differs from gate")
    current = load_json(files["current_season"], "B7 completed current season")
    _require(stage_evaluator["atlas_sha256"] == current.get("atlas_sha256"),
             "B7 stage evaluator Atlas differs from completed season")
    _require(stage_evaluator["atlas_catalog_sha256"] ==
             current.get("atlas_catalog_sha256") ==
             value.get("atlas_catalog_sha256"),
             "B7 stage evaluator Atlas catalog differs")
    completed = _mapping(
        stage_evaluator.get("completed_season"),
        "B7 stage evaluator completed season",
    )
    _require(completed.get("sha256") == _file_sha256(files["current_season"]),
             "B7 evaluator completed-season binding differs")
    _require(isinstance(completed.get("path"), str) and
             Path(completed["path"]).is_absolute(),
             "B7 evaluator completed-season path is not absolute")
    minimum = _integer(stage_evaluator["minimum_accepted_transitions"],
                       "B7 evaluator minimum transitions", minimum=1)
    counters = _mapping(current.get("counters"), "B7 completed counters")
    _require(_integer(counters.get("accepted_transitions"),
                      "B7 completed transitions", minimum=1) >= minimum,
             "B7 completed season misses evaluator transition minimum")
    predicates = stage_evaluator["predicates"]
    _require(isinstance(predicates, list) and predicates,
             "B7 stage evaluator predicates are empty")
    for index, predicate in enumerate(predicates):
        row = _mapping(predicate, f"B7 stage evaluator predicate {index}")
        _exact_keys(row, {"name", "path", "operator", "threshold", "observed", "passed"},
                    f"B7 stage evaluator predicate {index}")
        _text(row["name"], f"B7 stage evaluator predicate {index} name")
        _text(row["path"], f"B7 stage evaluator predicate {index} path")
        _text(row["operator"], f"B7 stage evaluator predicate {index} operator")
        _number(row["threshold"], f"B7 stage evaluator predicate {index} threshold")
        _number(row["observed"], f"B7 stage evaluator predicate {index} observed")
        _require(row["passed"] is True,
                 f"B7 stage evaluator predicate {row['name']} is red")
    stage_specific = _mapping(
        stage_evaluator["stage_specific"], "B7 stage-7 stage-specific evidence",
    )
    _exact_keys(stage_specific, {
        "guide_on_reference", "guide_on_completed_season", "guide_on_checkpoint",
        "guide_on_policy_version", "matched_seed", "guide_on_task_success",
        "guide_off_task_success", "degradation_fraction",
        "maximum_degradation_fraction", "global_dropout_rate", "passed",
    }, "B7 stage-7 stage-specific evidence")
    for name in (
        "guide_on_reference", "guide_on_completed_season", "guide_on_checkpoint",
    ):
        record = _mapping(stage_specific[name], f"B7 stage-7 {name}")
        _exact_keys(record, {"path", "sha256"}, f"B7 stage-7 {name}")
        _text(record["path"], f"B7 stage-7 {name} path")
        _digest(record["sha256"], f"B7 stage-7 {name} SHA-256")
    _integer(stage_specific["guide_on_policy_version"],
             "B7 stage-7 guide-on policy version", minimum=1)
    _integer(stage_specific["matched_seed"], "B7 stage-7 matched seed")
    guide_on = _number(stage_specific["guide_on_task_success"],
                       "B7 stage-7 guide-on success", 0.0, 1.0)
    guide_off = _number(stage_specific["guide_off_task_success"],
                        "B7 stage-7 guide-off success", 0.0, 1.0)
    degradation = _number(stage_specific["degradation_fraction"],
                          "B7 stage-7 degradation", 0.0, 1.0)
    maximum = _number(stage_specific["maximum_degradation_fraction"],
                      "B7 stage-7 maximum degradation", 0.0, 1.0)
    _number(stage_specific["global_dropout_rate"],
            "B7 stage-7 global dropout", 0.0, 1.0)
    _require(guide_on > 0.0 and abs(
        degradation - max(0.0, (guide_on - guide_off) / guide_on)
    ) <= 1e-12, "B7 stage-7 degradation derivation differs")
    _require(degradation <= maximum and stage_specific["passed"] is True,
             "B7 stage-7 stage-specific predicate is red")
    return digest


def _current_season(value: Mapping[str, Any]) -> str:
    digest = _validate_seal(value, CURRENT_SEASON_SCHEMA, "current season")
    _require(value.get("health") == "training-active", "current season is not active")
    _require(value.get("promotion_claim") is False,
             "current season made a forbidden promotion claim")
    _require(value.get("stage_id") == 7 and value.get("stage_name") ==
             "full-guide-off-ablation", "current season is not stage 7 guide-off")
    _digest(value.get("runtime_manifest_sha256"), "current season runtime")
    _digest(value.get("atlas_sha256"), "current season Atlas")
    _digest(value.get("atlas_catalog_sha256"), "current season Atlas catalog")
    _digest(value.get("lineage_root_sha256"), "current season lineage")
    _digest(value.get("causal_metrics_window_sha256"), "current causal metrics")
    _digest(value.get("network_metrics_window_sha256"), "current network metrics")
    causal = _mapping(value.get("causal_metrics_window"), "causal metrics window")
    network = _mapping(value.get("network_metrics_window"), "network metrics window")
    _require(canonical_sha256(causal) == value["causal_metrics_window_sha256"],
             "causal metrics window digest differs")
    _require(canonical_sha256(network) == value["network_metrics_window_sha256"],
             "network metrics window digest differs")
    counters = _mapping(value.get("counters"), "current season counters")
    accepted = _integer(counters.get("accepted_transitions"),
                        "current accepted transitions", minimum=1)
    _require(causal.get("accepted_transitions") == accepted,
             "causal metrics transition count differs")
    _require(network.get("network_client/transitions_accepted") == accepted,
             "network metrics transition count differs")
    coverage = _mapping(causal.get("observer_coverage"), "observer coverage")
    for name in (
        "missing_movement_command_samples", "missing_movement_speed_samples",
        "missing_true_view_pitch_samples", "missing_guide_dropout_samples",
    ):
        _require(coverage.get(name) == 0, f"observer coverage {name} is nonzero")
    _require(causal.get("private_causal_payload_serialized") is False,
             "season serialized private causal payload")
    privilege = _mapping(causal.get("privilege"), "season privilege metrics")
    _require(privilege.get("teacher_field_violations") == 0,
             "teacher-only fields reached the public trainer")
    return digest


def _predicate(failures: Sequence[str]) -> dict[str, Any]:
    return {"passed": not failures, "failures": list(failures)}


def _g0(*, archive: Mapping[str, Any], files: Mapping[str, Path],
        current: Mapping[str, Any], current_seal: str,
        evaluation_seal: str, b7_seal: str,
        legacy_seal: str) -> dict[str, Any]:
    """Seal the concrete identity/retirement checks completed before G1."""
    b6_role = INTEGRATION_EVIDENCE_ROLES["wsl_b6_campaign"]
    assertions = {
        "archive_exact_read_only": True,
        "runtime_manifest_attested": True,
        "integration_rerun_passed": True,
        "source_clean_and_b6_bound": True,
        "fresh_lineage_stage7_bound": True,
        "atlas_catalog_active_atlas_bound": True,
        "legacy_selectors_absent": True,
        "public_teacher_fields_absent": True,
        "private_causal_payload_absent": True,
    }
    evidence = {
        "archive_manifest_sha256": _digest(
            archive.get("manifest_sha256"), "G0 archive manifest seal",
        ),
        "runtime_manifest_sha256": _digest(
            current.get("runtime_manifest_sha256"), "G0 runtime manifest",
        ),
        "integration_report_sha256": _file_sha256(files["integration_report"]),
        "b6_campaign_sha256": _file_sha256(files[b6_role]),
        "source_identity_sha256": _file_sha256(files["source_identity"]),
        "retirement_manifest_sha256": _file_sha256(files["retirement_manifest"]),
        "atlas_catalog_sha256": _digest(
            current.get("atlas_catalog_sha256"), "G0 Atlas catalog",
        ),
        "b7_gate_evidence_sha256": _digest(b7_seal, "G0 B7 gate seal"),
        "legacy_absence_evidence_sha256": _digest(
            legacy_seal, "G0 legacy-absence seal",
        ),
        "current_season_evidence_sha256": _digest(
            current_seal, "G0 current-season seal",
        ),
        "evaluation_evidence_sha256": _digest(
            evaluation_seal, "G0 evaluation seal",
        ),
    }
    body = {
        "schema": G0_SUMMARY_SCHEMA,
        "assertions": assertions,
        "evidence": evidence,
    }
    summary = {**body, "evidence_sha256": canonical_sha256(body)}
    failures = [name for name, passed in assertions.items() if passed is not True]
    return {**_predicate(failures), "evidence": summary}


def _add(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _g1(current: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    counters = _mapping(current["counters"], "current counters")
    network = _mapping(current["network_metrics_window"], "network metrics")
    accepted = _integer(counters["accepted_transitions"], "G1 transitions")
    _add(accepted >= MIN_TRANSITIONS, failures,
         f"accepted transitions {accepted} < {MIN_TRANSITIONS}")
    _add(network.get("network_client/failed_rounds") == 0, failures,
         "failed rounds are nonzero")
    _add(network.get("network_client/echo_timeouts") == 0, failures,
         "echo timeouts are nonzero")
    accept_rate = _number(network.get("network_client/authoritative_echo_accept_rate"),
                          "G1 action echo rate", minimum=0.0, maximum=1.0)
    _add(accept_rate >= MIN_ACTION_ECHO_ACCEPTANCE, failures,
         f"action echo acceptance {accept_rate} < {MIN_ACTION_ECHO_ACCEPTANCE}")
    _add(_integer(network.get("network_client/map_epoch_resyncs"),
                  "G1 map epoch recoveries") >= 1, failures,
         "map-epoch recovery was not exercised")
    _add(_integer(network.get("network_client/telemetry_gap_resyncs"),
                  "G1 telemetry-gap recoveries") >= 1, failures,
         "whole-batch telemetry-gap recovery was not exercised")
    raw = _mapping(evaluation.get("g1"), "G1 evaluation")
    _exact_keys(raw, {
        "vertical_intent_echo_match_rate", "water_land_projection_skew",
        "partial_client_timeout_fatal_exercised",
    }, "G1 evaluation")
    vertical = _number(raw["vertical_intent_echo_match_rate"],
                       "G1 vertical echo rate", minimum=0.0, maximum=1.0)
    _add(vertical >= MIN_VERTICAL_ECHO_MATCH, failures,
         f"vertical echo match {vertical} < {MIN_VERTICAL_ECHO_MATCH}")
    _add(_integer(raw["water_land_projection_skew"], "G1 projection skew") == 0,
         failures, "water/land projection skew is nonzero")
    _add(raw["partial_client_timeout_fatal_exercised"] is True, failures,
         "partial-client timeout fatality was not exercised")
    return _predicate(failures)


def _g2(current: Mapping[str, Any], evaluation: Mapping[str, Any],
        stage_configuration_sha256: str) -> dict[str, Any]:
    failures: list[str] = []
    raw = _mapping(evaluation.get("g2"), "G2 evaluation")
    _exact_keys(raw, {
        "stage_configuration_sha256", "no_visible_target_samples",
        "moving_at_least_96_samples", "downlook_rate",
        "moving_mean_pitch_deg", "forward_command_rate",
        "backward_command_rate", "unnecessary_actual_crouch_rate",
        "unnecessary_actual_crouch_max", "action_collapse_audits",
    }, "G2 evaluation")
    _add(raw["stage_configuration_sha256"] == stage_configuration_sha256,
         failures, "G2 thresholds are not bound to the B7 stage configuration")
    _add(_integer(raw["no_visible_target_samples"], "G2 no-target samples") > 0,
         failures, "G2 no-visible-target subset is empty")
    _add(_integer(raw["moving_at_least_96_samples"], "G2 moving samples") > 0,
         failures, "G2 moving-at-least-96 subset is empty")
    downlook = _number(raw["downlook_rate"], "G2 down-look rate", 0.0, 1.0)
    pitch = _number(raw["moving_mean_pitch_deg"], "G2 moving pitch")
    forward = _number(raw["forward_command_rate"], "G2 forward rate", 0.0, 1.0)
    backward = _number(raw["backward_command_rate"], "G2 backward rate", 0.0, 1.0)
    crouch = _number(raw["unnecessary_actual_crouch_rate"],
                     "G2 unnecessary crouch rate", 0.0, 1.0)
    crouch_max = _number(raw["unnecessary_actual_crouch_max"],
                         "G2 frozen crouch maximum", 0.0, 1.0)
    _add(downlook <= MAX_DOWNLOOK_RATE, failures,
         f"down-look rate {downlook} > {MAX_DOWNLOOK_RATE}")
    _add(abs(pitch) <= MAX_MOVING_PITCH_DEG, failures,
         f"moving pitch {pitch} outside +/-{MAX_MOVING_PITCH_DEG}")
    _add(forward >= backward, failures, "forward command rate is below backward")
    _add(backward <= MAX_BACKWARD_RATE, failures,
         f"backward command rate {backward} > {MAX_BACKWARD_RATE}")
    _add(crouch < crouch_max, failures,
         "unnecessary actual-crouch rate did not stay below frozen threshold")
    audits = _mapping(raw["action_collapse_audits"], "G2 action collapse audits")
    _add(set(audits) == {"jump", "crouch", "strafe", "hook"}, failures,
         "G2 action-collapse audits do not cover jump/crouch/strafe/hook")
    for action in sorted(audits):
        audit = _mapping(audits[action], f"G2 {action} audit")
        _exact_keys(audit, {"observed_rate", "minimum_rate", "maximum_rate", "collapsed"},
                    f"G2 {action} audit")
        observed = _number(audit["observed_rate"], f"G2 {action} rate", 0.0, 1.0)
        lower = _number(audit["minimum_rate"], f"G2 {action} minimum", 0.0, 1.0)
        upper = _number(audit["maximum_rate"], f"G2 {action} maximum", 0.0, 1.0)
        _add(lower <= observed <= upper and lower <= upper, failures,
             f"{action} rate is outside the frozen stage interval")
        _add(audit["collapsed"] is False, failures, f"{action} action collapsed")

    # The complete observer window is a conservative second bar: a clean
    # no-target subset cannot hide a run-wide posture/motion collapse.
    causal = _mapping(current["causal_metrics_window"], "causal metrics")
    movement = _mapping(causal.get("movement"), "movement metrics")
    whole_down = _number(movement.get("downlook_over_15deg_rate"),
                         "whole-season down-look", 0.0, 1.0)
    whole_forward = _number(movement.get("forward_command_mean"),
                            "whole-season forward", 0.0)
    whole_backward = _number(movement.get("backward_command_mean"),
                             "whole-season backward", 0.0)
    _add(whole_down <= MAX_DOWNLOOK_RATE, failures,
         "whole-season down-look rate breaches G2")
    _add(whole_forward >= whole_backward, failures,
         "whole-season forward command mean is below backward")
    _add(whole_backward <= MAX_BACKWARD_RATE, failures,
         "whole-season backward command mean breaches G2")
    return _predicate(failures)


def _g3(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    raw = _mapping(evaluation.get("g3"), "G3 evaluation")
    _exact_keys(raw, {
        "section17_fixtures_passed", "stock_determinism_passed",
        "matched_seed_sha256", "baseline_episodes", "treatment_episodes",
        "baseline_safe_arrivals", "treatment_safe_arrivals",
        "baseline_environmental_deaths", "treatment_environmental_deaths",
        "progress_credit_cap_violations", "boundary_oscillation_credits",
        "invalid_hook_net_positive_events", "engine_invalid_anchor_recoveries",
    }, "G3 evaluation")
    _add(raw["section17_fixtures_passed"] is True, failures,
         "section-17 fixtures are not green")
    _add(raw["stock_determinism_passed"] is True, failures,
         "stock-map determinism is not green")
    _digest(raw["matched_seed_sha256"], "G3 matched seed digest")
    baseline_n = _integer(raw["baseline_episodes"], "G3 baseline episodes", minimum=1)
    treatment_n = _integer(raw["treatment_episodes"], "G3 treatment episodes", minimum=1)
    _add(baseline_n == treatment_n, failures, "G3 episode counts are not matched")
    base_safe = _integer(raw["baseline_safe_arrivals"], "G3 baseline safe arrivals")
    treat_safe = _integer(raw["treatment_safe_arrivals"], "G3 treatment safe arrivals")
    base_death = _integer(raw["baseline_environmental_deaths"], "G3 baseline deaths")
    treat_death = _integer(raw["treatment_environmental_deaths"], "G3 treatment deaths")
    _add(treat_safe / treatment_n > base_safe / baseline_n, failures,
         "G3 safe-arrival rate did not improve")
    _add(treat_death / treatment_n < base_death / baseline_n, failures,
         "G3 environmental-death rate did not improve")
    for name, message in (
        ("progress_credit_cap_violations", "progress-credit cap was violated"),
        ("boundary_oscillation_credits", "boundary oscillation farmed credit"),
        ("invalid_hook_net_positive_events", "invalid hook attempts were net-positive"),
        ("engine_invalid_anchor_recoveries", "recovery used an engine-invalid hook anchor"),
    ):
        _add(_integer(raw[name], f"G3 {name}") == 0, failures, message)
    return _predicate(failures)


def _g4(current: Mapping[str, Any], evaluation: Mapping[str, Any],
        stage_evaluator: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    raw = _mapping(evaluation.get("g4"), "G4 evaluation")
    _exact_keys(raw, {
        "matched_seed_sha256", "matched_task_set_sha256",
        "b7_stage_evaluation_evidence_sha256", "matched_attempts",
        "guide_on_successes", "guide_off_successes", "neutral_successes",
        "global_dropout_after_stage4_rate", "guide_off_downlook_rate",
        "guide_off_backward_rate", "guide_off_combat_events",
    }, "G4 evaluation")
    stage_seal = _validate_seal(
        stage_evaluator, "q2-multires-b7-stage-evaluation-v1",
        "G4 B7 stage evaluator",
    )
    _add(raw["b7_stage_evaluation_evidence_sha256"] == stage_seal, failures,
         "G4 binds a different B7 stage evaluator")
    stage_specific = _mapping(
        stage_evaluator.get("stage_specific"), "G4 B7 stage-specific evidence",
    )
    matched_seed = _integer(stage_specific.get("matched_seed"),
                            "G4 B7 matched seed")
    expected_seed_sha256 = canonical_sha256({
        "domain": "q2-multires-g4-matched-seed-v1",
        "matched_seed": matched_seed,
    })
    _add(raw["matched_seed_sha256"] == expected_seed_sha256, failures,
         "G4 matched seed differs from sealed B7 measurement evidence")
    _digest(raw["matched_task_set_sha256"], "G4 matched task digest")
    attempts = _integer(raw["matched_attempts"], "G4 matched attempts", minimum=1)
    on = _integer(raw["guide_on_successes"], "G4 guide-on successes")
    off = _integer(raw["guide_off_successes"], "G4 guide-off successes")
    neutral = _integer(raw["neutral_successes"], "G4 neutral successes")
    _add(on <= attempts and off <= attempts and neutral <= attempts, failures,
         "G4 successes exceed matched attempts")
    if on <= 0:
        failures.append("guide-on task success is zero")
    else:
        degradation = max(0.0, (on - off) / on)
        _add(degradation <= MAX_GUIDE_DEGRADATION, failures,
             f"guide-off degradation {degradation} > {MAX_GUIDE_DEGRADATION}")
    _add(off > neutral, failures, "guide-off policy does not exceed neutral baseline")
    _add(abs(on / attempts - _number(
        stage_specific.get("guide_on_task_success"),
        "G4 sealed guide-on success", 0.0, 1.0,
    )) <= 1e-12, failures,
         "G4 guide-on successes differ from sealed B7 measurement evidence")
    _add(abs(off / attempts - _number(
        stage_specific.get("guide_off_task_success"),
        "G4 sealed guide-off success", 0.0, 1.0,
    )) <= 1e-12, failures,
         "G4 guide-off successes differ from sealed B7 measurement evidence")
    dropout = _number(raw["global_dropout_after_stage4_rate"],
                      "G4 global dropout", 0.0, 1.0)
    _add(dropout > 0.0, failures, "global guide dropout was zero after stage 4")
    _add(abs(dropout - _number(
        stage_specific.get("global_dropout_rate"),
        "G4 sealed global dropout", 0.0, 1.0,
    )) <= 1e-12, failures,
         "G4 dropout differs from sealed B7 measurement evidence")
    causal = _mapping(current.get("causal_metrics_window"), "G4 causal metrics")
    guides = _mapping(causal.get("guides"), "G4 causal guide metrics")
    _add(abs(dropout - _number(
        guides.get("global_drop_rate"), "G4 causal global dropout", 0.0, 1.0,
    )) <= 1e-12, failures, "G4 dropout differs from the causal window")
    downlook = _number(raw["guide_off_downlook_rate"],
                       "G4 guide-off down-look", 0.0, 1.0)
    backward = _number(raw["guide_off_backward_rate"],
                       "G4 guide-off backward", 0.0, 1.0)
    movement = _mapping(causal.get("movement"), "G4 causal movement metrics")
    _add(abs(downlook - _number(
        movement.get("downlook_over_15deg_rate"),
        "G4 causal down-look", 0.0, 1.0,
    )) <= 1e-12, failures, "G4 down-look differs from the causal window")
    _add(abs(backward - _number(
        movement.get("backward_command_mean"),
        "G4 causal backward rate", 0.0, 1.0,
    )) <= 1e-12, failures, "G4 backward rate differs from the causal window")
    _add(downlook <= MAX_DOWNLOOK_RATE, failures,
         "guide-off reopened down-look collapse")
    _add(backward <= MAX_BACKWARD_RATE, failures,
         "guide-off reopened backward collapse")
    events = _mapping(raw["guide_off_combat_events"], "G4 guide-off combat events")
    _add(set(events) == {"actionable_exposure", "post_command_alignment",
                         "fire_permission", "executed_fire", "hits",
                         "repeated_hits", "kills"}, failures,
         "guide-off combat ladder fields differ")
    for name, value in events.items():
        _add(_integer(value, f"G4 guide-off {name}") > 0, failures,
             f"guide-off {name} is zero")
    combat = _mapping(causal.get("combat"), "G4 causal combat metrics")
    _add(dict(events) == {name: combat.get(name) for name in events}, failures,
         "G4 combat ladder differs from the causal window")
    return _predicate(failures)


def _g5(current: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    causal = _mapping(current["causal_metrics_window"], "causal metrics")
    combat = _mapping(causal.get("combat"), "combat metrics")
    names = (
        "actionable_exposure", "post_command_alignment", "fire_permission",
        "executed_fire", "hits", "repeated_hits", "kills",
    )
    values = [_integer(combat.get(name), f"G5 {name}") for name in names]
    for name, count in zip(names, values):
        _add(count > 0, failures, f"combat ladder {name} is zero")
    _add(all(left >= right for left, right in zip(values, values[1:])), failures,
         "combat ladder counts violate causal order")
    precision = _number(combat.get("aligned_fire_precision"),
                        "G5 aligned-fire precision", 0.0, 1.0)
    _add(precision >= MIN_ALIGNED_FIRE_PRECISION, failures,
         f"aligned-fire precision {precision} < {MIN_ALIGNED_FIRE_PRECISION}")
    _add(_integer(combat.get("hidden_fire"), "G5 hidden fire") == 0,
         failures, "hidden fire is nonzero")
    raw = _mapping(evaluation.get("g5"), "G5 evaluation")
    _exact_keys(raw, {
        "visible_contact_yaw_samples", "visible_contact_pitch_samples",
        "aligned_fire_precision_min", "visible_contact_yaw_mae_max_deg",
        "visible_contact_pitch_mae_max_deg",
    }, "G5 evaluation")
    _add(_integer(raw["visible_contact_yaw_samples"], "G5 yaw samples") > 0,
         failures, "visible-contact yaw sample set is empty")
    _add(_integer(raw["visible_contact_pitch_samples"], "G5 pitch samples") > 0,
         failures, "visible-contact pitch sample set is empty")
    precision_bar = _number(raw["aligned_fire_precision_min"],
                            "G5 frozen precision bar", 0.0, 1.0)
    yaw_bar = _number(raw["visible_contact_yaw_mae_max_deg"],
                      "G5 frozen yaw bar", 0.0)
    pitch_bar = _number(raw["visible_contact_pitch_mae_max_deg"],
                        "G5 frozen pitch bar", 0.0)
    _add(precision_bar >= MIN_ALIGNED_FIRE_PRECISION, failures,
         "frozen aligned-fire bar is weaker than 85%")
    _add(yaw_bar <= MAX_YAW_MAE_DEG, failures,
         "frozen yaw bar is weaker than 12 degrees")
    _add(pitch_bar <= MAX_PITCH_MAE_DEG, failures,
         "frozen pitch bar is weaker than 8 degrees")
    yaw = _number(combat.get("visible_contact_yaw_mae_deg"), "G5 yaw MAE", 0.0)
    pitch = _number(combat.get("visible_contact_pitch_mae_deg"), "G5 pitch MAE", 0.0)
    _add(precision >= precision_bar, failures, "aligned-fire precision misses frozen bar")
    _add(yaw <= yaw_bar, failures, f"visible-contact yaw MAE {yaw} misses frozen bar")
    _add(pitch <= pitch_bar, failures,
         f"visible-contact pitch MAE {pitch} misses frozen bar")
    return _predicate(failures)


def _artifact_bindings(evaluation: Mapping[str, Any], files: Mapping[str, Path]) -> None:
    bindings = _mapping(evaluation.get("artifact_sha256"), "evaluation artifact bindings")
    expected = set(SEASON_ROLES) - {"evaluation"}
    _require(set(bindings) == expected, "evaluation artifact bindings differ")
    for role in sorted(expected):
        _require(bindings[role] == _file_sha256(files[role]),
                 f"evaluation {role} binding differs from archive bytes")


def assemble_season(manifest_path: Path) -> dict[str, Any]:
    archive, files = validate_archive(manifest_path, kind="season")
    current = load_json(files["current_season"], "current season")
    current_seal = _current_season(current)
    evaluation = load_json(files["evaluation"], "season evaluation")
    evaluation_seal = _validate_seal(
        evaluation, SEASON_EVALUATION_SCHEMA, "season evaluation",
    )
    _exact_keys(evaluation, {
        "schema", "season_id", "season_kind", "current_season_evidence_sha256",
        "runtime_manifest_sha256", "atlas_sha256", "atlas_catalog_sha256",
        "lineage_root_sha256",
        "b7_gate_evidence_sha256", "artifact_sha256", "g1", "g2", "g3",
        "g4", "g5", "automatic_promotion", "evidence_sha256",
    }, "season evaluation")
    season_kind = archive["season_kind"]
    _require(evaluation["season_kind"] == season_kind,
             "evaluation and archive season kinds differ")
    causal = _mapping(current["causal_metrics_window"], "causal metrics")
    _require(evaluation["season_id"] == causal.get("season_id"),
             "evaluation and current season identities differ")
    _require(evaluation["current_season_evidence_sha256"] == current_seal,
             "evaluation binds a different current season")
    _require(evaluation["runtime_manifest_sha256"] == current["runtime_manifest_sha256"],
             "evaluation runtime differs from current season")
    _require(evaluation["atlas_sha256"] == current["atlas_sha256"],
             "evaluation Atlas differs from current season")
    _require(evaluation["atlas_catalog_sha256"] ==
             current["atlas_catalog_sha256"],
             "evaluation Atlas catalog differs from current season")
    _require(evaluation["lineage_root_sha256"] == current["lineage_root_sha256"],
             "evaluation lineage differs from current season")
    _require(evaluation["automatic_promotion"] is False,
             "season evaluator claims automatic promotion")
    _artifact_bindings(evaluation, files)
    _validate_identity_artifacts(files, current)

    b7 = load_json(files["b7_gate"], "B7 stage-7 gate")
    b7_seal = _validate_b7_gate(b7, files)
    stage_evaluator = load_json(
        files["b7_stage_evaluation"], "B7 stage-7 evaluator",
    )
    _require(evaluation["b7_gate_evidence_sha256"] == b7_seal,
             "season evaluator binds a different B7 gate")
    _require(b7["runtime_manifest_sha256"] == current["runtime_manifest_sha256"],
             "B7/current runtime identity differs")
    _require(b7["lineage_root_sha256"] == current["lineage_root_sha256"],
             "B7/current lineage identity differs")
    legacy = load_json(files["legacy_selector_absence"], "legacy absence")
    legacy_seal = _legacy_absent(legacy)

    predicates = {
        "G0": _g0(
            archive=archive, files=files, current=current,
            current_seal=current_seal, evaluation_seal=evaluation_seal,
            b7_seal=b7_seal, legacy_seal=legacy_seal,
        ),
        "G1": _g1(current, evaluation),
        "G2": _g2(current, evaluation, str(b7["stage_configuration_sha256"])),
        "G3": _g3(evaluation),
        "G4": _g4(current, evaluation, stage_evaluator),
        "G5": _g5(current, evaluation),
    }
    failures = [f"{name}: {failure}" for name, result in predicates.items()
                for failure in result["failures"]]
    gate = {
        "schema": SEASON_GATE_SCHEMA,
        "tool": TOOL,
        "season_id": evaluation["season_id"],
        "season_kind": season_kind,
        "archive": {
            "archive_id": archive["archive_id"],
            **_record(_regular(manifest_path, "archive manifest")),
            "manifest_sha256": archive["manifest_sha256"],
        },
        "identities": {
            "runtime_manifest_sha256": current["runtime_manifest_sha256"],
            "atlas_sha256": current["atlas_sha256"],
            "atlas_catalog_sha256": current["atlas_catalog_sha256"],
            "lineage_root_sha256": current["lineage_root_sha256"],
            "b7_gate_evidence_sha256": b7_seal,
            "legacy_absence_evidence_sha256": legacy_seal,
            "artifacts": _identity_records(files),
        },
        "inputs": {
            "current_season": _record(files["current_season"]),
            "evaluation": _record(files["evaluation"]),
            "current_season_evidence_sha256": current_seal,
            "evaluation_evidence_sha256": evaluation_seal,
        },
        "window": {
            "causal_metrics_window_sha256":
                current["causal_metrics_window_sha256"],
            "network_metrics_window_sha256":
                current["network_metrics_window_sha256"],
            "policy_start_version": causal["policy_start_version"],
            "policy_end_version": causal["policy_end_version"],
            "accepted_transitions": causal["accepted_transitions"],
        },
        "predicates": predicates,
        "failures": failures,
        "passed": not failures,
        "automatic_promotion": False,
        "public_mutation_authorized": False,
    }
    gate["gate_sha256"] = canonical_sha256({"domain": SEASON_GATE_SCHEMA, "gate": gate})
    _validate_checked_schema(gate, SEASON_GATE_SCHEMA, "assembled season gate")
    return gate


def _validate_gate_seal(value: Mapping[str, Any], schema: str, label: str) -> str:
    _require(value.get("schema") == schema, f"{label} schema differs")
    body = dict(value)
    digest = _digest(body.pop("gate_sha256", None), f"{label} gate seal")
    _require(digest == canonical_sha256({"domain": schema, "gate": body}),
             f"{label} gate seal differs")
    return digest


def _validate_g0_predicate(value: Any, label: str) -> None:
    predicate = _mapping(value, f"{label} G0")
    _exact_keys(predicate, {"passed", "failures", "evidence"}, f"{label} G0")
    _require(predicate["passed"] is True and predicate["failures"] == [],
             f"{label} G0 is red")
    summary = _mapping(predicate["evidence"], f"{label} G0 evidence")
    _validate_seal(summary, G0_SUMMARY_SCHEMA, f"{label} G0 evidence")
    _exact_keys(summary, {
        "schema", "assertions", "evidence", "evidence_sha256",
    }, f"{label} G0 evidence")
    assertions = _mapping(summary["assertions"], f"{label} G0 assertions")
    expected_assertions = {
        "archive_exact_read_only", "runtime_manifest_attested",
        "integration_rerun_passed", "source_clean_and_b6_bound",
        "fresh_lineage_stage7_bound", "atlas_catalog_active_atlas_bound",
        "legacy_selectors_absent", "public_teacher_fields_absent",
        "private_causal_payload_absent",
    }
    _exact_keys(assertions, expected_assertions, f"{label} G0 assertions")
    _require(all(assertions[name] is True for name in expected_assertions),
             f"{label} G0 assertion is not true")
    evidence = _mapping(summary["evidence"], f"{label} G0 bindings")
    expected_evidence = {
        "archive_manifest_sha256", "runtime_manifest_sha256",
        "integration_report_sha256", "b6_campaign_sha256",
        "source_identity_sha256", "retirement_manifest_sha256",
        "atlas_catalog_sha256", "b7_gate_evidence_sha256",
        "legacy_absence_evidence_sha256", "current_season_evidence_sha256",
        "evaluation_evidence_sha256",
    }
    _exact_keys(evidence, expected_evidence, f"{label} G0 bindings")
    for name in expected_evidence:
        _digest(evidence[name], f"{label} G0 {name}")


def _validate_season_gate(value: Mapping[str, Any], kind: str) -> str:
    digest = _validate_gate_seal(value, SEASON_GATE_SCHEMA, f"{kind} season gate")
    _require(value.get("season_kind") == kind, f"{kind} gate kind differs")
    _require(value.get("passed") is True and value.get("failures") == [],
             f"{kind} season gate is red")
    _require(value.get("automatic_promotion") is False and
             value.get("public_mutation_authorized") is False,
             f"{kind} season gate contains a mutation selector")
    predicates = _mapping(value.get("predicates"), f"{kind} predicates")
    _require(set(predicates) == {"G0", "G1", "G2", "G3", "G4", "G5"},
             f"{kind} predicate set differs")
    for name, result in predicates.items():
        predicate = _mapping(result, f"{kind} {name}")
        _require(predicate.get("passed") is True and predicate.get("failures") == [],
                 f"{kind} {name} is red")
    _validate_g0_predicate(predicates["G0"], kind)
    _text(value.get("season_id"), f"{kind} season id")
    archive = _mapping(value.get("archive"), f"{kind} archive identity")
    _exact_keys(archive, {"archive_id", "bytes", "sha256", "manifest_sha256"},
                f"{kind} archive identity")
    _text(archive["archive_id"], f"{kind} archive id")
    _integer(archive["bytes"], f"{kind} archive manifest bytes", minimum=1)
    _digest(archive["sha256"], f"{kind} archive manifest file SHA-256")
    _digest(archive["manifest_sha256"], f"{kind} archive manifest seal")
    inputs = _mapping(value.get("inputs"), f"{kind} season inputs")
    _exact_keys(inputs, {
        "current_season", "evaluation", "current_season_evidence_sha256",
        "evaluation_evidence_sha256",
    }, f"{kind} season inputs")
    for role in ("current_season", "evaluation"):
        record = _mapping(inputs[role], f"{kind} {role} record")
        _exact_keys(record, {"bytes", "sha256"}, f"{kind} {role} record")
        _integer(record["bytes"], f"{kind} {role} bytes", minimum=1)
        _digest(record["sha256"], f"{kind} {role} SHA-256")
    _digest(inputs["current_season_evidence_sha256"],
            f"{kind} current season evidence seal")
    _digest(inputs["evaluation_evidence_sha256"],
            f"{kind} evaluation evidence seal")
    window = _mapping(value.get("window"), f"{kind} season window")
    _exact_keys(window, {
        "causal_metrics_window_sha256", "network_metrics_window_sha256",
        "policy_start_version", "policy_end_version", "accepted_transitions",
    }, f"{kind} season window")
    _digest(window["causal_metrics_window_sha256"],
            f"{kind} causal metrics window seal")
    _digest(window["network_metrics_window_sha256"],
            f"{kind} network metrics window seal")
    start = _integer(window["policy_start_version"],
                     f"{kind} policy start version")
    end = _integer(window["policy_end_version"], f"{kind} policy end version")
    _require(end >= start, f"{kind} policy window is reversed")
    _integer(window["accepted_transitions"], f"{kind} window transitions",
             minimum=MIN_TRANSITIONS)
    _validate_checked_schema(value, SEASON_GATE_SCHEMA, f"{kind} season gate")
    return digest


def _season_identity(value: Mapping[str, Any], kind: str) -> dict[str, Any]:
    archive = _mapping(value["archive"], f"{kind} archive identity")
    inputs = _mapping(value["inputs"], f"{kind} season inputs")
    window = _mapping(value["window"], f"{kind} season window")
    identities = _mapping(value["identities"], f"{kind} identities")
    artifacts = _mapping(identities["artifacts"], f"{kind} artifact identities")
    atlas_artifact = _mapping(artifacts["atlas"], f"{kind} Atlas artifact")
    current_record = _mapping(inputs["current_season"],
                              f"{kind} current season record")
    evaluation_record = _mapping(inputs["evaluation"],
                                 f"{kind} evaluation record")
    return {
        "season_id": value["season_id"],
        "archive_id": archive["archive_id"],
        "archive_manifest_sha256": archive["manifest_sha256"],
        "archive_manifest_file_sha256": archive["sha256"],
        "current_season_file_sha256": current_record["sha256"],
        "evaluation_file_sha256": evaluation_record["sha256"],
        "current_season_evidence_sha256":
            inputs["current_season_evidence_sha256"],
        "evaluation_evidence_sha256": inputs["evaluation_evidence_sha256"],
        "causal_metrics_window_sha256":
            window["causal_metrics_window_sha256"],
        "network_metrics_window_sha256":
            window["network_metrics_window_sha256"],
        "atlas_sha256": identities["atlas_sha256"],
        "atlas_artifact_sha256": atlas_artifact["sha256"],
    }


def _independent_seasons(generated: Mapping[str, Any],
                         stock: Mapping[str, Any]) -> dict[str, Any]:
    """Return sealed inputs only after proving two separately observed seasons."""
    identities = {
        "generated": _season_identity(generated, "generated"),
        "stock": _season_identity(stock, "stock"),
    }
    for field in SEASON_DISTINCT_FIELDS:
        _require(identities["generated"][field] != identities["stock"][field],
                 f"generated/stock seasons reuse {field}")
    return identities


def _validate_independent_season_records(value: Any) -> None:
    records = _mapping(value, "B8 season independence")
    _exact_keys(records, {"generated", "stock"}, "B8 season independence")
    for kind in ("generated", "stock"):
        record = _mapping(records[kind], f"B8 {kind} season identity")
        _exact_keys(record, set(SEASON_DISTINCT_FIELDS),
                    f"B8 {kind} season identity")
        _text(record["season_id"], f"B8 {kind} season id")
        _text(record["archive_id"], f"B8 {kind} archive id")
        for field in SEASON_DISTINCT_FIELDS[2:]:
            _digest(record[field], f"B8 {kind} {field}")
    for field in SEASON_DISTINCT_FIELDS:
        _require(records["generated"][field] != records["stock"][field],
                 f"B8 generated/stock seasons reuse {field}")


def _validate_shadow(evidence: Mapping[str, Any], files: Mapping[str, Path],
                     expected_identity: Mapping[str, Any]) -> str:
    digest = _validate_seal(evidence, SHADOW_EVIDENCE_SCHEMA, "shadow evidence")
    _exact_keys(evidence, {
        "schema", "shadow_id", "season_gate_sha256", "artifact_sha256",
        "topology", "vps_work", "rotation", "events", "transport",
        "legacy_absence_evidence_sha256", "shadow_only",
        "automatic_promotion", "public_mutations", "evidence_sha256",
    }, "shadow evidence")
    gates = _mapping(evidence["season_gate_sha256"], "shadow season gates")
    _require(set(gates) == {"generated", "stock"}, "shadow season gate set differs")
    bindings = _mapping(evidence["artifact_sha256"], "shadow artifact bindings")
    _require(set(bindings) == set(SHADOW_ROLES) - {"shadow_evidence"},
             "shadow artifact bindings differ")
    for role in bindings:
        _require(bindings[role] == _file_sha256(files[role]),
                 f"shadow {role} binding differs")
    actual_identity = _identity_records(files)
    _same_identity(actual_identity, expected_identity, "shadow/season")
    legacy = load_json(files["legacy_selector_absence"], "shadow legacy absence")
    legacy_seal = _legacy_absent(legacy)
    _require(evidence["legacy_absence_evidence_sha256"] == legacy_seal,
             "shadow binds a different legacy-absence report")
    _require(evidence["shadow_only"] is True and
             evidence["automatic_promotion"] is False and
             evidence["public_mutations"] == [],
             "shadow evidence contains a deployment/mutation selector")
    return digest


def _g6(evidence: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    topology = _mapping(evidence["topology"], "shadow topology")
    _exact_keys(topology, {"maxclients", "ml_clients", "human_slots"},
                "shadow topology")
    _add(topology["maxclients"] == MAXCLIENTS, failures, "maxclients is not 6")
    _add(topology["ml_clients"] == CLIENTS, failures, "ML client count is not 4")
    _add(topology["human_slots"] == HUMAN_SLOTS, failures, "human slots are not 2")
    vps = _mapping(evidence["vps_work"], "shadow VPS work")
    _exact_keys(vps, {"compile_processes", "analyze_processes", "compile_launches",
                      "analyze_launches"}, "shadow VPS work")
    for name, value in vps.items():
        _add(_integer(value, f"shadow {name}") == 0, failures,
             f"VPS {name} is nonzero")
    rotation = _mapping(evidence["rotation"], "shadow rotation")
    _exact_keys(rotation, {"stock_maps", "generated_maps", "interlaced",
                           "queue_prefix_isolation"}, "shadow rotation")
    _add(_integer(rotation["stock_maps"], "shadow stock maps") > 0,
         failures, "shadow did not load a stock map")
    _add(_integer(rotation["generated_maps"], "shadow generated maps") > 0,
         failures, "shadow did not load a generated map")
    _add(rotation["interlaced"] is True, failures, "shadow rotation was not interlaced")
    _add(rotation["queue_prefix_isolation"] is True, failures,
         "shadow queue prefix isolation failed")
    events = _mapping(evidence["events"], "shadow events")
    _exact_keys(events, {"human_joins", "human_leaves", "map_changes"},
                "shadow events")
    for name in events:
        _add(_integer(events[name], f"shadow {name}") >= 1, failures,
             f"shadow {name} was not exercised")
    transport = _mapping(evidence["transport"], "shadow transport")
    _exact_keys(transport, {
        "accepted_transitions", "failed_rounds", "echo_timeouts",
        "authoritative_echo_accept_rate", "vertical_intent_echo_match_rate",
        "water_land_projection_skew", "map_epoch_recoveries",
        "telemetry_gap_recoveries", "partial_client_timeout_remained_fatal",
    }, "shadow transport")
    _add(_integer(transport["accepted_transitions"], "shadow transitions") >= MIN_TRANSITIONS,
         failures, f"shadow accepted fewer than {MIN_TRANSITIONS} transitions")
    _add(transport["failed_rounds"] == 0, failures, "shadow failed rounds are nonzero")
    _add(transport["echo_timeouts"] == 0, failures, "shadow echo timeouts are nonzero")
    _add(_number(transport["authoritative_echo_accept_rate"],
                 "shadow echo acceptance", 0.0, 1.0) >= MIN_ACTION_ECHO_ACCEPTANCE,
         failures, "shadow action echo acceptance breaches G1")
    _add(_number(transport["vertical_intent_echo_match_rate"],
                 "shadow vertical echo", 0.0, 1.0) >= MIN_VERTICAL_ECHO_MATCH,
         failures, "shadow vertical echo breaches G1")
    _add(_integer(transport["water_land_projection_skew"], "shadow skew") == 0,
         failures, "shadow water/land projection skew is nonzero")
    _add(_integer(transport["map_epoch_recoveries"], "shadow map recoveries") >= 1,
         failures, "shadow map-epoch recovery was not exercised")
    _add(_integer(transport["telemetry_gap_recoveries"], "shadow gap recoveries") >= 1,
         failures, "shadow telemetry-gap recovery was not exercised")
    _add(transport["partial_client_timeout_remained_fatal"] is True, failures,
         "partial-client timeout was relaxed in shadow")
    return _predicate(failures)


def assemble_b8(generated_path: Path, stock_path: Path,
                shadow_manifest_path: Path) -> dict[str, Any]:
    generated = load_json(generated_path, "generated season gate")
    stock = load_json(stock_path, "stock season gate")
    generated_seal = _validate_season_gate(generated, "generated")
    stock_seal = _validate_season_gate(stock, "stock")
    season_independence = _independent_seasons(generated, stock)
    gen_identity = _mapping(_mapping(generated["identities"], "generated identities")
                            .get("artifacts"), "generated artifact identities")
    stock_identity = _mapping(_mapping(stock["identities"], "stock identities")
                              .get("artifacts"), "stock artifact identities")
    _same_identity(gen_identity, stock_identity, "generated/stock")
    _require(generated["identities"]["runtime_manifest_sha256"] ==
             stock["identities"]["runtime_manifest_sha256"],
             "generated/stock runtime semantic identity differs")
    _require(generated["identities"]["lineage_root_sha256"] ==
             stock["identities"]["lineage_root_sha256"],
             "generated/stock policy lineage differs")
    _require(generated["identities"]["atlas_catalog_sha256"] ==
             stock["identities"]["atlas_catalog_sha256"],
             "generated/stock Atlas catalog identity differs")

    archive, files = validate_archive(shadow_manifest_path, kind="shadow")
    _validate_identity_artifacts(files)
    shadow = load_json(files["shadow_evidence"], "shadow evidence")
    shadow_seal = _validate_shadow(shadow, files, gen_identity)
    _require(shadow["season_gate_sha256"] == {
        "generated": generated_seal, "stock": stock_seal,
    }, "shadow binds different season gates")
    g6 = _g6(shadow)
    predicates = {
        "G0": generated["predicates"]["G0"],
        "G1_generated": generated["predicates"]["G1"],
        "G1_stock": stock["predicates"]["G1"],
        "G2_generated": generated["predicates"]["G2"],
        "G2_stock": stock["predicates"]["G2"],
        "G3_generated": generated["predicates"]["G3"],
        "G3_stock": stock["predicates"]["G3"],
        "G4_generated": generated["predicates"]["G4"],
        "G4_stock": stock["predicates"]["G4"],
        "G5_generated": generated["predicates"]["G5"],
        "G5_stock": stock["predicates"]["G5"],
        "G6_shadow": g6,
    }
    failures = [f"{name}: {failure}" for name, result in predicates.items()
                for failure in result["failures"]]
    gate = {
        "schema": B8_GATE_SCHEMA,
        "tool": TOOL,
        "inputs": {
            "generated_gate": {**_record(generated_path), "gate_sha256": generated_seal},
            "stock_gate": {**_record(stock_path), "gate_sha256": stock_seal},
            "shadow_archive": {**_record(shadow_manifest_path),
                               "manifest_sha256": archive["manifest_sha256"]},
            "shadow_evidence": {**_record(files["shadow_evidence"]),
                                "evidence_sha256": shadow_seal},
        },
        "identities": {
            "runtime_manifest_sha256": generated["identities"]["runtime_manifest_sha256"],
            "atlas_catalog_sha256": generated["identities"]["atlas_catalog_sha256"],
            "lineage_root_sha256": generated["identities"]["lineage_root_sha256"],
            "artifacts": {
                role: gen_identity[role] for role in STABLE_IDENTITY_ROLES
            },
            "atlases": {
                "generated": {
                    "semantic_sha256": generated["identities"]["atlas_sha256"],
                    "artifact": gen_identity["atlas"],
                },
                "stock": {
                    "semantic_sha256": stock["identities"]["atlas_sha256"],
                    "artifact": stock_identity["atlas"],
                },
                "shadow": _record(files["atlas"]),
            },
        },
        "season_independence": season_independence,
        "predicates": predicates,
        "failures": failures,
        "passed": not failures,
        "cold_restart_pending": True,
        "automatic_promotion": False,
        "public_mutation_authorized": False,
    }
    gate["gate_sha256"] = canonical_sha256({"domain": B8_GATE_SCHEMA, "gate": gate})
    _validate_checked_schema(gate, B8_GATE_SCHEMA, "assembled B8 gate")
    return gate


def _validate_b8_gate(value: Mapping[str, Any]) -> str:
    digest = _validate_gate_seal(value, B8_GATE_SCHEMA, "B8 gate")
    _require(value.get("passed") is True and value.get("failures") == [],
             "B8 gate is red")
    _require(value.get("cold_restart_pending") is True,
             "B8 gate does not preserve the B9 cold-restart boundary")
    _require(value.get("automatic_promotion") is False and
             value.get("public_mutation_authorized") is False,
             "B8 gate contains a public mutation selector")
    predicates = _mapping(value.get("predicates"), "B8 predicates")
    expected = {
        "G0", "G1_generated", "G1_stock", "G2_generated", "G2_stock",
        "G3_generated", "G3_stock", "G4_generated", "G4_stock",
        "G5_generated", "G5_stock", "G6_shadow",
    }
    _require(set(predicates) == expected, "B8 predicate set differs")
    for name, result in predicates.items():
        predicate = _mapping(result, f"B8 {name}")
        _require(predicate.get("passed") is True and predicate.get("failures") == [],
                 f"B8 {name} is red")
    _validate_independent_season_records(value.get("season_independence"))
    _validate_g0_predicate(predicates["G0"], "B8")
    _validate_checked_schema(value, B8_GATE_SCHEMA, "B8 gate")
    return digest


def _validate_reconstruction_manifest(
    value: Mapping[str, Any], files: Mapping[str, Path],
    b8: Mapping[str, Any], b8_seal: str,
) -> str:
    digest = _validate_seal(
        value, RECONSTRUCTION_MANIFEST_SCHEMA, "reconstruction manifest",
        "manifest_sha256",
    )
    _validate_checked_schema(
        value, RECONSTRUCTION_MANIFEST_SCHEMA, "reconstruction manifest",
    )
    _exact_keys(value, {
        "schema", "reconstruction_id", "b8_gate_sha256",
        "runtime_manifest_sha256", "atlas_catalog_sha256",
        "lineage_root_sha256", "artifacts", "complete", "manifest_sha256",
    }, "reconstruction manifest")
    _text(value["reconstruction_id"], "reconstruction id")
    _require(value["b8_gate_sha256"] == b8_seal,
             "reconstruction manifest binds a different B8 gate")
    identities = _mapping(b8.get("identities"), "B8 identities")
    for field in (
        "runtime_manifest_sha256", "atlas_catalog_sha256",
        "lineage_root_sha256",
    ):
        _require(value[field] == identities.get(field),
                 f"reconstruction manifest {field} differs from B8")
    artifacts = _mapping(value["artifacts"], "reconstruction artifacts")
    _require(set(artifacts) == set(IDENTITY_ROLES),
             "reconstruction artifact inventory differs")
    for role in IDENTITY_ROLES:
        binding = _mapping(artifacts[role], f"reconstruction {role} binding")
        _exact_keys(binding, {"bytes", "sha256"},
                    f"reconstruction {role} binding")
        _require(binding == _record(files[role]),
                 f"reconstruction {role} binding differs from archive bytes")
    expected_identity = _mapping(
        identities.get("artifacts"), "B8 artifact identities",
    )
    _same_identity(_identity_records(files), expected_identity,
                   "reconstruction/B8")
    atlases = _mapping(identities.get("atlases"), "B8 Atlas identities")
    _require(_record(files["atlas"]) == atlases.get("shadow"),
             "reconstruction Atlas inventory differs from B8 shadow")
    _require(value["complete"] is True,
             "reconstruction manifest is not complete")
    return digest


def _validate_cold_restart(evidence: Mapping[str, Any], files: Mapping[str, Path],
                           b8: Mapping[str, Any], b8_seal: str) -> tuple[
                               str, dict[str, Any], str
                           ]:
    digest = _validate_seal(evidence, COLD_RESTART_SCHEMA, "cold restart evidence")
    _exact_keys(evidence, {
        "schema", "restart_id", "b8_gate_sha256", "artifact_sha256",
        "shadow_stopped_cleanly", "reconstructed_from_attested_archive",
        "reconstruction_manifest_sha256", "server_reconnected",
        "ml_clients_reconnected", "policy_loaded", "stock_maps_loaded",
        "generated_maps_loaded", "map_changes", "accepted_transitions",
        "failed_rounds", "echo_timeouts", "legacy_absence_evidence_sha256",
        "operational_legacy_selector_matches", "automatic_promotion",
        "public_mutations", "evidence_sha256",
    }, "cold restart evidence")
    _require(evidence["b8_gate_sha256"] == b8_seal,
             "cold restart binds a different B8 gate")
    bindings = _mapping(evidence["artifact_sha256"], "cold restart bindings")
    _require(set(bindings) == set(RESTART_ROLES) - {"cold_restart_evidence"},
             "cold restart artifact bindings differ")
    for role in bindings:
        _require(bindings[role] == _file_sha256(files[role]),
                 f"cold restart {role} binding differs")
    b8_identity = _mapping(_mapping(b8["identities"], "B8 identities").get("artifacts"),
                           "B8 artifact identities")
    _same_identity(_identity_records(files), b8_identity, "cold-restart/B8")
    atlases = _mapping(_mapping(b8["identities"], "B8 identities").get("atlases"),
                       "B8 Atlas identities")
    _require(_record(files["atlas"]) == atlases.get("shadow"),
             "cold restart Atlas inventory differs from shadow")
    reconstruction = load_json(
        files["reconstruction_manifest"], "reconstruction manifest",
    )
    reconstruction_seal = _validate_reconstruction_manifest(
        reconstruction, files, b8, b8_seal,
    )
    _require(evidence["reconstruction_manifest_sha256"] ==
             _file_sha256(files["reconstruction_manifest"]),
             "cold restart reconstruction manifest digest differs")
    legacy = load_json(files["legacy_selector_absence"], "restart legacy absence")
    legacy_seal = _legacy_absent(legacy)
    _require(evidence["legacy_absence_evidence_sha256"] == legacy_seal,
             "cold restart binds a different legacy-absence report")
    failures: list[str] = []
    for name in (
        "shadow_stopped_cleanly", "reconstructed_from_attested_archive",
        "server_reconnected", "policy_loaded",
    ):
        _add(evidence[name] is True, failures, f"cold restart {name} is not true")
    _add(evidence["ml_clients_reconnected"] == CLIENTS, failures,
         "cold restart did not reconnect four ML clients")
    _add(_integer(evidence["stock_maps_loaded"], "restart stock maps") >= 1,
         failures, "cold restart did not load a stock map")
    _add(_integer(evidence["generated_maps_loaded"], "restart generated maps") >= 1,
         failures, "cold restart did not load a generated map")
    _add(_integer(evidence["map_changes"], "restart map changes") >= 2,
         failures, "cold restart did not prove map rotation")
    _add(_integer(evidence["accepted_transitions"], "restart transitions") >= CLIENTS,
         failures, "cold restart admitted no full four-client round")
    _add(evidence["failed_rounds"] == 0, failures,
         "cold restart failed rounds are nonzero")
    _add(evidence["echo_timeouts"] == 0, failures,
         "cold restart echo timeouts are nonzero")
    _add(evidence["operational_legacy_selector_matches"] == [], failures,
         "cold restart found operational legacy selectors")
    _add(evidence["automatic_promotion"] is False and
         evidence["public_mutations"] == [], failures,
         "cold restart evidence contains a public mutation selector")
    return digest, _predicate(failures), reconstruction_seal


def assemble_b9(b8_path: Path, restart_manifest_path: Path) -> dict[str, Any]:
    b8 = load_json(b8_path, "B8 gate")
    b8_seal = _validate_b8_gate(b8)
    archive, files = validate_archive(restart_manifest_path, kind="restart")
    _validate_identity_artifacts(files)
    _require(_file_sha256(files["b8_gate"]) == _file_sha256(_regular(b8_path, "B8 gate")),
             "restart archive B8 gate bytes differ from selected B8 gate")
    archived_b8 = load_json(files["b8_gate"], "archived B8 gate")
    _require(archived_b8 == b8, "restart archive B8 gate content differs")
    cold = load_json(files["cold_restart_evidence"], "cold restart evidence")
    cold_seal, cold_predicate, reconstruction_seal = _validate_cold_restart(
        cold, files, b8, b8_seal,
    )
    reconstruction_sha256 = _file_sha256(files["reconstruction_manifest"])
    predicates = dict(_mapping(b8["predicates"], "B8 predicates"))
    predicates["G6_cold_restart"] = cold_predicate
    failures = [f"{name}: {failure}" for name, result in predicates.items()
                for failure in result["failures"]]
    eligible = not failures
    promotion_manifest = {
        "runtime_manifest_sha256": b8["identities"]["runtime_manifest_sha256"],
        "atlas_catalog_sha256": b8["identities"]["atlas_catalog_sha256"],
        "atlases": b8["identities"]["atlases"],
        "lineage_root_sha256": b8["identities"]["lineage_root_sha256"],
        "artifact_identities": b8["identities"]["artifacts"],
        "b8_gate_sha256": b8_seal,
        "cold_restart_evidence_sha256": cold_seal,
        "reconstruction_manifest_sha256": reconstruction_sha256,
        "reconstruction_manifest_evidence_sha256": reconstruction_seal,
        "complete_predicate": "G0&&G1&&G2&&G3&&G4&&G5_generated&&G5_stock&&G6",
    }
    promotion_manifest["manifest_sha256"] = canonical_sha256({
        "domain": "q2-multires-b9-promotion-manifest-v1",
        "manifest": promotion_manifest,
    })
    decision = {
        "schema": B9_DECISION_SCHEMA,
        "tool": TOOL,
        "inputs": {
            "b8_gate": {**_record(b8_path), "gate_sha256": b8_seal},
            "restart_archive": {**_record(restart_manifest_path),
                                "manifest_sha256": archive["manifest_sha256"]},
            "cold_restart_evidence": {**_record(files["cold_restart_evidence"]),
                                      "evidence_sha256": cold_seal},
            "reconstruction_manifest": {
                **_record(files["reconstruction_manifest"]),
                "manifest_sha256": reconstruction_seal,
            },
        },
        "promotion_manifest": promotion_manifest,
        "gate_comparison": predicates,
        "failures": failures,
        "decision": "eligible-for-root-manual-promotion" if eligible else "hold",
        "eligible": eligible,
        "automatic_promotion": False,
        "public_mutation_performed": False,
        "public_mutation_authorized": False,
        "root_manual_promotion_required": eligible,
    }
    decision["decision_sha256"] = canonical_sha256({
        "domain": B9_DECISION_SCHEMA, "decision": decision,
    })
    _validate_checked_schema(decision, B9_DECISION_SCHEMA,
                             "assembled B9 decision")
    return decision


def validate_document(path: Path) -> dict[str, Any]:
    value = load_json(path, "gate document")
    schema = value.get("schema")
    if schema in (SEASON_GATE_SCHEMA, B8_GATE_SCHEMA, B9_DECISION_SCHEMA):
        _validate_checked_schema(value, str(schema), "gate document")
    if schema == SEASON_GATE_SCHEMA:
        kind = value.get("season_kind")
        _require(kind in ("generated", "stock"), "season gate kind is invalid")
        _validate_season_gate(value, str(kind))
    elif schema == B8_GATE_SCHEMA:
        _validate_b8_gate(value)
    elif schema == B9_DECISION_SCHEMA:
        _exact_keys(value, {
            "schema", "tool", "inputs", "promotion_manifest",
            "gate_comparison", "failures", "decision", "eligible",
            "automatic_promotion", "public_mutation_performed",
            "public_mutation_authorized", "root_manual_promotion_required",
            "decision_sha256",
        }, "B9 decision")
        body = dict(value)
        digest = _digest(body.pop("decision_sha256", None), "B9 decision seal")
        _require(digest == canonical_sha256({"domain": schema, "decision": body}),
                 "B9 decision seal differs")
        manifest = dict(_mapping(value.get("promotion_manifest"), "promotion manifest"))
        _exact_keys(manifest, {
            "runtime_manifest_sha256", "atlas_catalog_sha256", "atlases",
            "lineage_root_sha256", "artifact_identities", "b8_gate_sha256",
            "cold_restart_evidence_sha256", "reconstruction_manifest_sha256",
            "reconstruction_manifest_evidence_sha256", "complete_predicate",
            "manifest_sha256",
        }, "promotion manifest")
        manifest_digest = _digest(manifest.pop("manifest_sha256", None),
                                  "promotion manifest seal")
        _require(manifest_digest == canonical_sha256({
            "domain": "q2-multires-b9-promotion-manifest-v1",
            "manifest": manifest,
        }), "promotion manifest seal differs")
        inputs = _mapping(value.get("inputs"), "B9 inputs")
        _exact_keys(inputs, {
            "b8_gate", "restart_archive", "cold_restart_evidence",
            "reconstruction_manifest",
        }, "B9 inputs")
        for role, seal_name in (
            ("b8_gate", "gate_sha256"),
            ("restart_archive", "manifest_sha256"),
            ("cold_restart_evidence", "evidence_sha256"),
            ("reconstruction_manifest", "manifest_sha256"),
        ):
            record = _mapping(inputs[role], f"B9 {role} input")
            _exact_keys(record, {"bytes", "sha256", seal_name},
                        f"B9 {role} input")
            _integer(record["bytes"], f"B9 {role} bytes", minimum=1)
            _digest(record["sha256"], f"B9 {role} SHA-256")
            _digest(record[seal_name], f"B9 {role} {seal_name}")
        _require(inputs["b8_gate"]["gate_sha256"] ==
                 manifest["b8_gate_sha256"],
                 "B9 input/promotion B8 gate differs")
        _require(inputs["cold_restart_evidence"]["evidence_sha256"] ==
                 manifest["cold_restart_evidence_sha256"],
                 "B9 input/promotion cold-restart evidence differs")
        _require(inputs["reconstruction_manifest"]["sha256"] ==
                 manifest["reconstruction_manifest_sha256"] and
                 inputs["reconstruction_manifest"]["manifest_sha256"] ==
                 manifest["reconstruction_manifest_evidence_sha256"],
                 "B9 input/promotion reconstruction manifest differs")
        comparisons = _mapping(value.get("gate_comparison"), "B9 comparisons")
        _require(set(comparisons) == {
            "G0", "G1_generated", "G1_stock", "G2_generated", "G2_stock",
            "G3_generated", "G3_stock", "G4_generated", "G4_stock",
            "G5_generated", "G5_stock", "G6_shadow", "G6_cold_restart",
        }, "B9 comparison set differs")
        _validate_g0_predicate(comparisons["G0"], "B9")
        eligible = value.get("eligible") is True
        _require(value.get("automatic_promotion") is False and
                 value.get("public_mutation_performed") is False and
                 value.get("public_mutation_authorized") is False,
                 "B9 decision contains a mutation selector")
        _require(value.get("decision") == (
            "eligible-for-root-manual-promotion" if eligible else "hold"
        ), "B9 decision/eligibility differs")
        _require(value.get("root_manual_promotion_required") is eligible,
                 "B9 root-promotion boundary differs")
        all_green = all(
            isinstance(result, Mapping) and result.get("passed") is True and
            result.get("failures") == [] for result in comparisons.values()
        )
        _require(eligible is all_green and (value.get("failures") == []) is eligible,
                 "B9 eligibility does not equal the complete predicate")
    else:
        raise GateError(f"unsupported gate document schema {schema!r}")
    return value


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(path.expanduser()))
    _require(destination.is_absolute(), "output path must be absolute")
    _require(not destination.exists() and not destination.is_symlink(),
             "exclusive output path already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    archive = commands.add_parser(
        "archive", help="author one new exact read-only evidence archive",
    )
    archive.add_argument("--kind", choices=("season", "shadow", "restart"),
                         required=True)
    archive.add_argument("--season-kind", choices=("generated", "stock"))
    archive.add_argument("--archive-id", required=True)
    archive.add_argument(
        "--artifact", action="append", default=[], metavar="ROLE=/absolute/path",
        help="repeat for every role; integration evidence is discovered from its envelope",
    )
    archive.add_argument("--out-dir", type=Path, required=True)
    season = commands.add_parser("season", help="evaluate one immutable quality season")
    season.add_argument("--archive-manifest", type=Path, required=True)
    season.add_argument("--out", type=Path, required=True)
    b8 = commands.add_parser("b8", help="join generated, stock, and shadow evidence")
    b8.add_argument("--generated-gate", type=Path, required=True)
    b8.add_argument("--stock-gate", type=Path, required=True)
    b8.add_argument("--shadow-archive-manifest", type=Path, required=True)
    b8.add_argument("--out", type=Path, required=True)
    b9 = commands.add_parser("b9", help="emit the non-mutating final decision")
    b9.add_argument("--b8-gate", type=Path, required=True)
    b9.add_argument("--restart-archive-manifest", type=Path, required=True)
    b9.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate one emitted gate/decision")
    validate.add_argument("--document", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "archive":
            manifest = author_archive(
                destination=args.out_dir, kind=args.kind,
                archive_id=args.archive_id, season_kind=args.season_kind,
                artifacts=_artifact_specs(args.artifact),
            )
            print(manifest)
            return 0
        if args.command == "season":
            result = assemble_season(args.archive_manifest)
            _publish(args.out, result)
            return 0 if result["passed"] else 2
        if args.command == "b8":
            result = assemble_b8(
                args.generated_gate, args.stock_gate, args.shadow_archive_manifest,
            )
            _publish(args.out, result)
            return 0 if result["passed"] else 2
        if args.command == "b9":
            result = assemble_b9(args.b8_gate, args.restart_archive_manifest)
            _publish(args.out, result)
            return 0 if result["eligible"] else 2
        validate_document(args.document)
        return 0
    except (GateError, OSError) as error:
        print(f"{TOOL} failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
