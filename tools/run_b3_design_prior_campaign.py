#!/usr/bin/env python3
"""Prepare and evaluate the deterministic B3 design-prior campaign.

The producer has two deliberately separate phases. ``prepare`` freezes the
stock prior, paired seeds, generator treatment, implementation identity, and
regression thresholds before either generated lane exists. ``evaluate`` then
requires the complete baseline and treatment artifact sets and recomputes all
metrics from their compiled-Atlas design signatures.  It never discovers a
passing subset and it never treats source metadata as compiled-world proof.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_source_closure import (  # noqa: E402
    atlas_analyzer_authority_inputs,
    atlas_analyzer_authority_sha256,
)


PLAN_SCHEMA = "q2-b3-design-prior-plan-v1"
LANE_SCHEMA = "q2-b3-design-prior-lane-v1"
BIAS_SCHEMA = "q2-b3-generator-bias-v1"
SEEDS_SCHEMA = "q2-b3-design-prior-seeds-v1"
CAMPAIGN_SCHEMA = "q2-b3-design-prior-campaign-v1"
DESIGN_SIGNATURE_SCHEMA = "q2-atlas-design-signature-v1"
STOCK_MAPS = tuple(f"q2dm{number}" for number in range(1, 9))
STYLES = (
    "open", "towers", "canyon", "pits", "arena_open",
    "arena_vertical", "arena_lanes",
)
EDGE_TYPES = (
    "walk", "strafe_walk", "step", "jump", "controlled_drop",
    "crouch_enter", "crouch_hold", "crouch_exit", "water_transition",
    "mover", "teleporter", "hook", "ladder",
)
PROBABILITY_KNOBS = (
    "occupied_density", "corridor_prob", "hallway_ratio", "tower_prob",
    "lane_prob", "lava_prob", "extra_arena_prob", "large_building_ratio",
)
INTEGER_KNOBS = ("terrace_levels",)
RANGE_KNOBS = ("arena_cover_range", "corner_range")
ALLOWED_KNOBS = frozenset((*PROBABILITY_KNOBS, *INTEGER_KNOBS, *RANGE_KNOBS))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MAP_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
ITEM_CLASS = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PLACEHOLDER_MARKERS = ("placeholder", "synthetic", "dummy", "tbd", "todo", "fixme")

# A fixed, nearly uniform baseline.  The first style receives the indivisible
# remainder so the integer probabilities sum to exactly one million.
BASELINE_STYLE_WEIGHTS_PPM = {
    style: 142_857 + (1 if index == 0 else 0)
    for index, style in enumerate(STYLES)
}

DEFAULT_KNOBS = {
    "open": {
        "occupied_density": 550_000, "corridor_prob": 350_000,
        "hallway_ratio": 0, "terrace_levels": 1, "tower_prob": 150_000,
        "lane_prob": 350_000, "lava_prob": 100_000,
        "extra_arena_prob": 0, "arena_cover_range": [2, 4],
        "corner_range": [1, 1], "large_building_ratio": 0,
    },
    "towers": {
        "occupied_density": 550_000, "corridor_prob": 350_000,
        "hallway_ratio": 0, "terrace_levels": 2, "tower_prob": 900_000,
        "lane_prob": 350_000, "lava_prob": 150_000,
        "extra_arena_prob": 0, "arena_cover_range": [2, 4],
        "corner_range": [1, 1], "large_building_ratio": 0,
    },
    "canyon": {
        "occupied_density": 550_000, "corridor_prob": 350_000,
        "hallway_ratio": 0, "terrace_levels": 1, "tower_prob": 200_000,
        "lane_prob": 900_000, "lava_prob": 150_000,
        "extra_arena_prob": 0, "arena_cover_range": [2, 4],
        "corner_range": [1, 1], "large_building_ratio": 0,
    },
    "pits": {
        "occupied_density": 550_000, "corridor_prob": 350_000,
        "hallway_ratio": 0, "terrace_levels": 3, "tower_prob": 300_000,
        "lane_prob": 300_000, "lava_prob": 600_000,
        "extra_arena_prob": 0, "arena_cover_range": [2, 4],
        "corner_range": [1, 1], "large_building_ratio": 0,
    },
    "arena_open": {
        "occupied_density": 480_000, "corridor_prob": 180_000,
        "hallway_ratio": 180_000, "terrace_levels": 1,
        "tower_prob": 50_000, "lane_prob": 150_000,
        "lava_prob": 50_000, "extra_arena_prob": 450_000,
        "arena_cover_range": [4, 7], "corner_range": [2, 3],
        "large_building_ratio": 250_000,
    },
    "arena_vertical": {
        "occupied_density": 520_000, "corridor_prob": 200_000,
        "hallway_ratio": 180_000, "terrace_levels": 3,
        "tower_prob": 750_000, "lane_prob": 300_000,
        "lava_prob": 100_000, "extra_arena_prob": 400_000,
        "arena_cover_range": [3, 5], "corner_range": [1, 2],
        "large_building_ratio": 250_000,
    },
    "arena_lanes": {
        "occupied_density": 520_000, "corridor_prob": 180_000,
        "hallway_ratio": 180_000, "terrace_levels": 1,
        "tower_prob": 150_000, "lane_prob": 1_000_000,
        "lava_prob": 50_000, "extra_arena_prob": 500_000,
        "arena_cover_range": [3, 6], "corner_range": [2, 3],
        "large_building_ratio": 300_000,
    },
}

SCALAR_HISTOGRAMS = {
    "edges_per_node_milli": (0, 500, 1_000, 2_000, 4_000, 8_000, 16_000),
    "items_per_spawn_milli": (0, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000),
    "lightmapped_faces_ppm": (0, 100_000, 250_000, 500_000, 750_000, 900_000, 1_000_001),
}
METRIC_CONTRACT = {
    "algorithm": "integer-total-variation-ppm-v1",
    "primary_metrics": [
        "topology_distance_ppm", "economy_distance_ppm",
        "environment_distance_ppm",
    ],
    "minimum_improvement_ppm_each": 1,
    "minimum_aggregate_improvement_ppm": 1,
    "static_pass_rate_may_decrease": False,
    "required_pair_count": 28,
    "required_unique_layout_ratio_ppm": 1_000_000,
    "minimum_style_simpson_ppm": 700_000,
    "maximum_style_simpson_regression_ppm": 50_000,
    "minimum_descriptor_unique_ratio_ppm": 750_000,
}


class B3PriorError(ValueError):
    """Raised before publication for malformed or inadmissible evidence."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=True,
            separators=(",", ":"), sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise B3PriorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise B3PriorError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path, *, canonical: bool = True) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                B3PriorError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B3PriorError(f"cannot load {path}: {exc}") from exc
    if canonical and raw != canonical_bytes(value):
        raise B3PriorError(f"JSON is not canonical compact sorted JSON plus LF: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B3PriorError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise B3PriorError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise B3PriorError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: object, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise B3PriorError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise B3PriorError(f"{label} must be <= {maximum}")
    return value


def _signed_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise B3PriorError(f"{label} must be an integer")
    return value


def _digest(value: object, label: str, *, git: bool = False) -> str:
    pattern = HEX40 if git else HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise B3PriorError(f"{label} is not a lowercase {'Git' if git else 'SHA-256'} digest")
    if set(value) == {"0"}:
        raise B3PriorError(f"{label} is an all-zero placeholder digest")
    return value


def _reject_placeholder(value: object, label: str) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
            raise B3PriorError(f"{label} contains a placeholder/synthetic marker")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_placeholder(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholder(child, f"{label}[{index}]")


def _exclusive_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise B3PriorError(f"output already exists or cannot be created: {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def repository_binding(repo_root: Path = ROOT) -> dict[str, Any]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise B3PriorError(f"cannot bind repository identity: {exc}") from exc
    if status:
        raise B3PriorError("repository is not clean; refusing final B3 evidence")
    _digest(commit, "repository commit", git=True)
    _digest(tree, "repository tree", git=True)
    analyzer_inputs = atlas_analyzer_authority_inputs(repo_root)
    return {
        "repository_commit": commit,
        "repository_tree": tree,
        "git_clean": True,
        "generator_sha256": file_sha256(repo_root / "maps/generator.py"),
        "analyzer_authority_sha256": atlas_analyzer_authority_sha256(repo_root),
        "analyzer_authority_file_count": len(analyzer_inputs),
    }


def validate_implementation(value: object) -> dict[str, Any]:
    implementation = _mapping(value, "implementation")
    _exact_keys(
        implementation,
        {
            "repository_commit", "repository_tree", "git_clean",
            "generator_sha256", "analyzer_authority_sha256",
            "analyzer_authority_file_count",
        },
        "implementation",
    )
    _digest(implementation["repository_commit"], "repository commit", git=True)
    _digest(implementation["repository_tree"], "repository tree", git=True)
    if implementation["git_clean"] is not True:
        raise B3PriorError("implementation is not clean")
    _digest(implementation["generator_sha256"], "generator SHA-256")
    _digest(implementation["analyzer_authority_sha256"], "analyzer authority SHA-256")
    _integer(implementation["analyzer_authority_file_count"], "analyzer input count", 1)
    return dict(implementation)


def validate_design_signature(value: object, label: str) -> dict[str, Any]:
    signature = _mapping(value, label)
    _exact_keys(
        signature,
        {
            "schema", "coordinate_free", "bsp_sha256", "counts",
            "degree_histogram", "edge_type_histogram", "item_class_multiset",
            "light",
        },
        label,
    )
    if signature["schema"] != DESIGN_SIGNATURE_SCHEMA or signature["coordinate_free"] is not True:
        raise B3PriorError(f"{label} is not a coordinate-free Atlas design signature")
    _digest(signature["bsp_sha256"], f"{label} BSP SHA-256")
    counts = _mapping(signature["counts"], f"{label} counts")
    _exact_keys(
        counts,
        {"l1_nodes", "l1_edges", "deathmatch_spawns", "items", "faces", "visibility_clusters"},
        f"{label} counts",
    )
    for key in counts:
        _integer(counts[key], f"{label} counts.{key}", 0)
    if counts["l1_nodes"] < 1 or counts["deathmatch_spawns"] < 2 or counts["faces"] < 1:
        raise B3PriorError(f"{label} has vacuous map counts")
    degree = _mapping(signature["degree_histogram"], f"{label} degree histogram")
    for key, count in degree.items():
        if key not in {str(index) for index in range(16)}:
            raise B3PriorError(f"{label} has an invalid degree bin {key!r}")
        _integer(count, f"{label} degree[{key}]", 0)
    if sum(degree.values()) < 1:
        raise B3PriorError(f"{label} degree histogram is empty")
    edges = _mapping(signature["edge_type_histogram"], f"{label} edge histogram")
    for key, count in edges.items():
        if key not in EDGE_TYPES:
            raise B3PriorError(f"{label} has unknown edge type {key!r}")
        _integer(count, f"{label} edge[{key}]", 0)
    if sum(edges.values()) != counts["l1_edges"]:
        raise B3PriorError(f"{label} edge histogram does not equal l1_edges")
    items = _mapping(signature["item_class_multiset"], f"{label} item multiset")
    for key, count in items.items():
        if not isinstance(key, str) or not ITEM_CLASS.fullmatch(key):
            raise B3PriorError(f"{label} has invalid item class {key!r}")
        _integer(count, f"{label} item[{key}]", 0)
    if sum(items.values()) != counts["items"]:
        raise B3PriorError(f"{label} item multiset does not equal item count")
    light = _mapping(signature["light"], f"{label} light")
    _exact_keys(light, {"lightdata_bytes", "lightmapped_faces"}, f"{label} light")
    _integer(light["lightdata_bytes"], f"{label} lightdata bytes", 1)
    _integer(light["lightmapped_faces"], f"{label} lightmapped faces", 1)
    if light["lightmapped_faces"] > counts["faces"]:
        raise B3PriorError(f"{label} has more lightmapped faces than faces")
    _reject_placeholder(signature, label)
    return dict(signature)


def _ratio(numerator: int, denominator: int, scale: int) -> int:
    if denominator <= 0:
        raise B3PriorError("ratio denominator is not positive")
    return numerator * scale // denominator


def _scalar_bin(value: int, boundaries: Sequence[int]) -> str:
    for index in range(len(boundaries) - 1):
        if boundaries[index] <= value < boundaries[index + 1]:
            return f"{boundaries[index]}:{boundaries[index + 1]}"
    return f"{boundaries[-1]}:+inf"


def prior_pack(signatures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not signatures:
        raise B3PriorError("cannot build a prior from no signatures")
    item_bins = sorted({key for signature in signatures for key in signature["item_class_multiset"]})
    degree = {str(index): 0 for index in range(16)}
    edges = {name: 0 for name in EDGE_TYPES}
    items = {name: 0 for name in item_bins}
    scalar = {
        name: {
            _scalar_bin(boundaries[index], boundaries): 0
            for index in range(len(boundaries) - 1)
        }
        | {f"{boundaries[-1]}:+inf": 0}
        for name, boundaries in SCALAR_HISTOGRAMS.items()
    }
    source_hashes = []
    for signature in signatures:
        source_hashes.append(signature["bsp_sha256"])
        for key, value in signature["degree_histogram"].items():
            degree[key] += value
        for key, value in signature["edge_type_histogram"].items():
            edges[key] += value
        for key, value in signature["item_class_multiset"].items():
            items[key] += value
        counts = signature["counts"]
        light = signature["light"]
        values = {
            "edges_per_node_milli": _ratio(counts["l1_edges"], counts["l1_nodes"], 1_000),
            "items_per_spawn_milli": _ratio(counts["items"], counts["deathmatch_spawns"], 1_000),
            "lightmapped_faces_ppm": _ratio(light["lightmapped_faces"], counts["faces"], 1_000_000),
        }
        for name, value in values.items():
            scalar[name][_scalar_bin(value, SCALAR_HISTOGRAMS[name])] += 1
    pack = {
        "schema": "q2-b3-coordinate-free-prior-pack-v1",
        "coordinate_free": True,
        "map_count": len(signatures),
        "source_bsp_sha256": sorted(source_hashes),
        "histograms": {
            "degree_mass": degree,
            "edge_type_mass": edges,
            "item_class_mass": items,
            **scalar,
        },
    }
    pack["content_sha256"] = sha256_bytes(canonical_bytes(pack))
    return pack


def validate_prior_pack(value: object, label: str) -> dict[str, Any]:
    pack = _mapping(value, label)
    _exact_keys(
        pack,
        {
            "schema", "coordinate_free", "map_count", "source_bsp_sha256",
            "histograms", "content_sha256",
        },
        label,
    )
    if pack["schema"] != "q2-b3-coordinate-free-prior-pack-v1" or pack["coordinate_free"] is not True:
        raise B3PriorError(f"{label} schema/coordinate-free marker differs")
    map_count = _integer(pack["map_count"], f"{label} map count", 1)
    sources = _list(pack["source_bsp_sha256"], f"{label} source BSP hashes")
    if len(sources) != map_count or sources != sorted(sources) or len(set(sources)) != map_count:
        raise B3PriorError(f"{label} source BSP identities are not exact/distinct")
    for index, digest in enumerate(sources):
        _digest(digest, f"{label} source BSP {index}")
    histograms = _mapping(pack["histograms"], f"{label} histograms")
    _exact_keys(
        histograms,
        {"degree_mass", "edge_type_mass", "item_class_mass", *SCALAR_HISTOGRAMS},
        f"{label} histograms",
    )
    for name, raw in histograms.items():
        histogram = _mapping(raw, f"{label} {name}")
        if not histogram:
            raise B3PriorError(f"{label} {name} is empty")
        for key, count in histogram.items():
            if not isinstance(key, str) or not key:
                raise B3PriorError(f"{label} {name} has a noncanonical bin")
            _integer(count, f"{label} {name}[{key}]", 0)
        if sum(histogram.values()) < 1:
            raise B3PriorError(f"{label} {name} has zero total mass")
    recorded = _digest(pack["content_sha256"], f"{label} content SHA-256")
    unsigned = {key: child for key, child in pack.items() if key != "content_sha256"}
    if recorded != sha256_bytes(canonical_bytes(unsigned)):
        raise B3PriorError(f"{label} content SHA-256 does not bind the pack")
    return dict(pack)


def load_stock_prior(stock_analysis_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    for map_name in STOCK_MAPS:
        path = stock_analysis_dir / f"{map_name}.design-signature.json"
        signature = validate_design_signature(load_json(path, canonical=False), f"stock {map_name}")
        records[map_name] = {
            "file": {"bytes": path.stat().st_size, "sha256": file_sha256(path)},
            "bsp_sha256": signature["bsp_sha256"],
            "signature": signature,
        }
    hashes = [record["bsp_sha256"] for record in records.values()]
    if len(set(hashes)) != len(STOCK_MAPS):
        raise B3PriorError("stock design signatures do not bind eight distinct BSPs")
    return prior_pack([records[name]["signature"] for name in STOCK_MAPS]), records


def validate_bias(value: object) -> dict[str, Any]:
    bias = _mapping(value, "generator bias")
    _exact_keys(bias, {"schema", "style_weights_ppm", "knob_delta"}, "generator bias")
    if bias["schema"] != BIAS_SCHEMA:
        raise B3PriorError("generator bias schema differs")
    weights = _mapping(bias["style_weights_ppm"], "style weights")
    _exact_keys(weights, set(STYLES), "style weights")
    for style, value in weights.items():
        _integer(value, f"style weight {style}", 50_000, 300_000)
    if sum(weights.values()) != 1_000_000:
        raise B3PriorError("style weights must sum to exactly 1,000,000 ppm")
    if dict(weights) != BASELINE_STYLE_WEIGHTS_PPM:
        raise B3PriorError(
            "the B3 prototype freezes the same balanced style mixture in both lanes; "
            "treatment may alter only allowlisted generator knobs"
        )
    delta = _mapping(bias["knob_delta"], "knob delta")
    if not delta or not set(delta).issubset(ALLOWED_KNOBS):
        raise B3PriorError("knob delta must be a nonempty subset of the frozen allowlist")
    for key, value in delta.items():
        if key in PROBABILITY_KNOBS:
            _integer(value, f"{key} encoded delta", 0, 400_000)
            # Signed integer JSON is encoded without a bool; shift by 200k so
            # schema-level minimums remain simple and portable.
        elif key in INTEGER_KNOBS:
            _integer(value, f"{key} encoded delta", 0, 2)
        else:
            pair = _list(value, f"{key} encoded delta")
            if len(pair) != 2:
                raise B3PriorError(f"{key} encoded delta must have two elements")
            for index, element in enumerate(pair):
                _integer(element, f"{key}[{index}] encoded delta", 0, 4)
    if (
        dict(weights) == BASELINE_STYLE_WEIGHTS_PPM
        and all(
            (_decoded_delta(key, encoded) == 0)
            if key not in RANGE_KNOBS
            else (_decoded_delta(key, encoded) == [0, 0])
            for key, encoded in delta.items()
        )
    ):
        raise B3PriorError("treatment must change at least one allowlisted generator knob")
    _reject_placeholder(bias, "generator bias")
    return json.loads(json.dumps(bias))


def _decoded_delta(key: str, encoded: object) -> object:
    if key in PROBABILITY_KNOBS:
        return int(encoded) - 200_000
    if key in INTEGER_KNOBS:
        return int(encoded) - 1
    return [int(encoded[0]) - 2, int(encoded[1]) - 2]  # type: ignore[index]


def _effective_knobs(style: str, bias: Mapping[str, Any] | None) -> dict[str, Any]:
    result = json.loads(json.dumps(DEFAULT_KNOBS[style]))
    if bias is None:
        return result
    for key, encoded in bias["knob_delta"].items():
        delta = _decoded_delta(key, encoded)
        if key in PROBABILITY_KNOBS:
            result[key] += delta
            _integer(result[key], f"effective {style}.{key}", 0, 1_000_000)
        elif key in INTEGER_KNOBS:
            result[key] += delta
            _integer(result[key], f"effective {style}.{key}", 1, 4)
        else:
            result[key] = [result[key][index] + delta[index] for index in range(2)]
            if not (0 <= result[key][0] <= result[key][1] <= 12):
                raise B3PriorError(f"effective {style}.{key} is outside [0,12] or reversed")
    return result


def validate_seeds(value: object) -> list[int]:
    document = _mapping(value, "seed declaration")
    _exact_keys(document, {"schema", "seeds"}, "seed declaration")
    if document["schema"] != SEEDS_SCHEMA:
        raise B3PriorError("seed declaration schema differs")
    seeds = _list(document["seeds"], "seeds")
    if len(seeds) != METRIC_CONTRACT["required_pair_count"]:
        raise B3PriorError("the B3 campaign requires exactly 28 declared seed pairs")
    checked = [_integer(seed, f"seed[{index}]", 0, 2**31 - 1) for index, seed in enumerate(seeds)]
    if len(set(checked)) != len(checked):
        raise B3PriorError("campaign seeds must be unique")
    return checked


def prepare_campaign(
    campaign_id: str,
    stock_analysis_dir: Path,
    bias_path: Path,
    seeds_path: Path,
    output_path: Path,
    *,
    repo_root: Path = ROOT,
    _implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not MAP_ID.fullmatch(campaign_id):
        raise B3PriorError("campaign id is not canonical")
    implementation = validate_implementation(
        _implementation if _implementation is not None else repository_binding(repo_root)
    )
    bias = validate_bias(load_json(bias_path))
    seeds = validate_seeds(load_json(seeds_path))
    stock_prior, stock_records = load_stock_prior(stock_analysis_dir)
    normative = {
        "design_sha256": file_sha256(repo_root / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"),
        "plan_sha256": file_sha256(repo_root / "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md"),
    }
    rows = []
    for ordinal, seed in enumerate(seeds):
        # Four maps per explicit style lets the campaign reuse the already
        # qualified all-or-nothing B2 lifecycle without a bespoke compiler or
        # analyzer admission path. Both lanes retain the identical style.
        baseline_style = STYLES[ordinal // 4]
        treatment_style = baseline_style
        rows.append({
            "ordinal": ordinal,
            "seed": seed,
            "baseline": {
                "map": f"{campaign_id}_b_{ordinal:02d}",
                "style": baseline_style,
                "generator_knobs": _effective_knobs(baseline_style, None),
            },
            "treatment": {
                "map": f"{campaign_id}_t_{ordinal:02d}",
                "style": treatment_style,
                "generator_knobs": _effective_knobs(treatment_style, bias),
            },
        })
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "status": "frozen-before-generation",
        "community_input_used": False,
        "implementation": implementation,
        "normative_documents": normative,
        "stock_prior": stock_prior,
        "stock_signatures": {
            name: {
                "file": stock_records[name]["file"],
                "bsp_sha256": stock_records[name]["bsp_sha256"],
            }
            for name in STOCK_MAPS
        },
        "generator_bias": bias,
        "baseline_style_weights_ppm": BASELINE_STYLE_WEIGHTS_PPM,
        "metric_contract": METRIC_CONTRACT,
        "pairs": rows,
    }
    _exclusive_write(output_path, plan)
    return plan


def _histogram_with_other(
    histogram: Mapping[str, Any], bins: Sequence[str],
) -> dict[str, int]:
    result = {key: int(histogram.get(key, 0)) for key in bins}
    result["__other__"] = sum(int(value) for key, value in histogram.items() if key not in result)
    return result


def _tv_ppm(left: Mapping[str, int], right: Mapping[str, int]) -> int:
    keys = sorted(set(left) | set(right))
    left_total = sum(left.get(key, 0) for key in keys)
    right_total = sum(right.get(key, 0) for key in keys)
    if left_total <= 0 or right_total <= 0:
        raise B3PriorError("histogram distance cannot consume an empty distribution")
    numerator = sum(
        abs(left.get(key, 0) * right_total - right.get(key, 0) * left_total)
        for key in keys
    )
    return numerator * 1_000_000 // (2 * left_total * right_total)


def histogram_distances(stock: Mapping[str, Any], lane: Mapping[str, Any]) -> dict[str, int]:
    stock_hist = stock["histograms"]
    lane_hist = lane["histograms"]
    degree = _tv_ppm(stock_hist["degree_mass"], lane_hist["degree_mass"])
    edges = _tv_ppm(stock_hist["edge_type_mass"], lane_hist["edge_type_mass"])
    item_bins = list(stock_hist["item_class_mass"])
    items = _tv_ppm(
        _histogram_with_other(stock_hist["item_class_mass"], item_bins),
        _histogram_with_other(lane_hist["item_class_mass"], item_bins),
    )
    ratios = [
        _tv_ppm(stock_hist[name], lane_hist[name])
        for name in SCALAR_HISTOGRAMS
    ]
    result = {
        "topology_distance_ppm": (degree + edges) // 2,
        "economy_distance_ppm": items,
        "environment_distance_ppm": sum(ratios) // len(ratios),
    }
    result["aggregate_distance_ppm"] = sum(result.values()) // len(result)
    return result


def _descriptor_sha256(signature: Mapping[str, Any]) -> str:
    descriptor = {
        key: value for key, value in signature.items()
        if key not in {"bsp_sha256", "schema", "coordinate_free"}
    }
    return sha256_bytes(canonical_bytes(descriptor))


def _simpson_ppm(styles: Sequence[str]) -> int:
    if len(styles) < 2:
        return 0
    counts = Counter(styles)
    same_ordered_pairs = sum(count * (count - 1) for count in counts.values())
    return 1_000_000 - same_ordered_pairs * 1_000_000 // (len(styles) * (len(styles) - 1))


def _validate_lane(
    value: object,
    lane_name: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    analysis_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lane = _mapping(value, f"{lane_name} lane")
    _exact_keys(
        lane,
        {
            "schema", "campaign_id", "lane", "plan_sha256", "evidence_kind",
            "synthetic_claims", "implementation", "authorities", "pipeline",
            "maps", "failures", "passed",
        },
        f"{lane_name} lane",
    )
    if lane["schema"] != LANE_SCHEMA or lane["campaign_id"] != plan["campaign_id"]:
        raise B3PriorError(f"{lane_name} lane schema or campaign identity differs")
    _digest(lane["plan_sha256"], f"{lane_name} plan SHA-256")
    if lane["lane"] != lane_name or lane["plan_sha256"] != plan_sha256:
        raise B3PriorError(f"{lane_name} lane is not bound to the exact frozen plan")
    if lane["evidence_kind"] != "measured-compiled-atlas" or lane["synthetic_claims"] is not False:
        raise B3PriorError(f"{lane_name} lane is placeholder/synthetic evidence")
    if validate_implementation(lane["implementation"]) != plan["implementation"]:
        raise B3PriorError(f"{lane_name} lane implementation identity drifted")
    authorities = _mapping(lane["authorities"], f"{lane_name} authorities")
    _exact_keys(
        authorities,
        {
            "q2tool", "packer", "verifier", "cm_oracle", "pmove_oracle",
            "hook_oracle", "fall_oracle", "hook_attestation", "b1_gate",
        },
        f"{lane_name} authorities",
    )
    for name, raw in authorities.items():
        record = _mapping(raw, f"{lane_name} authority {name}")
        _exact_keys(record, {"bytes", "sha256"}, f"{lane_name} authority {name}")
        _integer(record["bytes"], f"{lane_name} authority {name} bytes", 1)
        _digest(record["sha256"], f"{lane_name} authority {name} SHA-256")
    pipeline = _mapping(lane["pipeline"], f"{lane_name} pipeline")
    expected_pipeline = {
        "declaration", "source_freeze", "compile", "compiled_cm_preflight",
        "materialization", "claims_prepare", "atlas_build", "claims_validation",
    }
    _exact_keys(pipeline, expected_pipeline, f"{lane_name} pipeline")
    for stage, raw in pipeline.items():
        record = _mapping(raw, f"{lane_name} pipeline {stage}")
        _exact_keys(record, {"bytes", "sha256"}, f"{lane_name} pipeline {stage}")
        _integer(record["bytes"], f"{lane_name} pipeline {stage} bytes", 1)
        _digest(record["sha256"], f"{lane_name} pipeline {stage} SHA-256")
    if lane["failures"] != [] or lane["passed"] is not True:
        raise B3PriorError(f"{lane_name} lane is not a complete passing artifact set")
    rows = _list(lane["maps"], f"{lane_name} maps")
    if len(rows) != len(plan["pairs"]):
        raise B3PriorError(f"{lane_name} lane does not contain every declared pair")
    signatures = []
    verified_rows = []
    layouts: set[str] = set()
    for expected_pair, raw in zip(plan["pairs"], rows):
        expected = expected_pair[lane_name]
        row = _mapping(raw, f"{lane_name} map row")
        _exact_keys(
            row,
            {
                "ordinal", "map", "seed", "style", "generator_knobs",
                "source_static_passed", "layout_sha256", "bsp_sha256",
                "design_signature",
            },
            f"{lane_name} map row",
        )
        row_ordinal = _integer(row["ordinal"], f"{lane_name} row ordinal", 0, 27)
        row_seed = _integer(row["seed"], f"{lane_name} row seed", 0, 2**31 - 1)
        if row_ordinal != expected_pair["ordinal"] or row_seed != expected_pair["seed"]:
            raise B3PriorError(f"{lane_name} lane seed/ordinal pairing differs")
        for key in ("map", "style", "generator_knobs"):
            if canonical_bytes(row[key]) != canonical_bytes(expected[key]):
                raise B3PriorError(f"{lane_name} row {row.get('ordinal')} {key} differs from plan")
        if row["source_static_passed"] is not True:
            raise B3PriorError(f"{lane_name} row {row['ordinal']} failed source/static validation")
        layout = _digest(row["layout_sha256"], f"{lane_name} layout SHA-256")
        bsp = _digest(row["bsp_sha256"], f"{lane_name} BSP SHA-256")
        artifact = _mapping(row["design_signature"], f"{lane_name} design artifact")
        _exact_keys(artifact, {"bytes", "sha256"}, f"{lane_name} design artifact")
        _integer(artifact["bytes"], f"{lane_name} design artifact bytes", 1)
        _digest(artifact["sha256"], f"{lane_name} design artifact SHA-256")
        path = analysis_dir / f"{row['map']}.design-signature.json"
        if not path.is_file() or path.is_symlink():
            raise B3PriorError(f"{lane_name} design signature is missing or a symlink: {path}")
        if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]:
            raise B3PriorError(f"{lane_name} design signature artifact identity drifted")
        signature = validate_design_signature(load_json(path, canonical=False), f"{lane_name} {row['map']}")
        if signature["bsp_sha256"] != bsp:
            raise B3PriorError(f"{lane_name} design signature BSP identity drifted")
        layouts.add(layout)
        signatures.append(signature)
        verified_rows.append({
            "ordinal": row["ordinal"], "seed": row["seed"], "map": row["map"],
            "style": row["style"], "layout_sha256": layout, "bsp_sha256": bsp,
            "design_signature": dict(artifact),
            "descriptor_sha256": _descriptor_sha256(signature),
        })
    if len(layouts) != len(rows):
        raise B3PriorError(f"{lane_name} lane contains duplicate exact layouts")
    pack = prior_pack(signatures)
    return verified_rows, pack


def validate_plan(value: object) -> dict[str, Any]:
    plan = _mapping(value, "campaign plan")
    _exact_keys(
        plan,
        {
            "schema", "campaign_id", "status", "community_input_used",
            "implementation", "normative_documents", "stock_prior",
            "stock_signatures", "generator_bias", "baseline_style_weights_ppm",
            "metric_contract", "pairs",
        },
        "campaign plan",
    )
    if plan["schema"] != PLAN_SCHEMA or plan["status"] != "frozen-before-generation":
        raise B3PriorError("campaign plan schema/status differs")
    if not isinstance(plan["campaign_id"], str) or not MAP_ID.fullmatch(plan["campaign_id"]):
        raise B3PriorError("campaign plan id is not canonical")
    if plan["community_input_used"] is not False:
        raise B3PriorError("community input is forbidden in the B3 prototype")
    validate_implementation(plan["implementation"])
    normative = _mapping(plan["normative_documents"], "normative documents")
    _exact_keys(normative, {"design_sha256", "plan_sha256"}, "normative documents")
    _digest(normative["design_sha256"], "design document SHA-256")
    _digest(normative["plan_sha256"], "plan document SHA-256")
    if plan["baseline_style_weights_ppm"] != BASELINE_STYLE_WEIGHTS_PPM:
        raise B3PriorError("baseline style mixture differs from the frozen contract")
    if canonical_bytes(plan["metric_contract"]) != canonical_bytes(METRIC_CONTRACT):
        raise B3PriorError("campaign metric contract differs from frozen constants")
    bias = validate_bias(plan["generator_bias"])
    prior = validate_prior_pack(plan["stock_prior"], "stock prior")
    if prior["map_count"] != len(STOCK_MAPS):
        raise B3PriorError("stock prior does not contain q2dm1 through q2dm8")
    stock = _mapping(plan["stock_signatures"], "stock signatures")
    _exact_keys(stock, set(STOCK_MAPS), "stock signatures")
    stock_bsp_hashes = []
    for map_name, raw in stock.items():
        record = _mapping(raw, f"stock signature {map_name}")
        _exact_keys(record, {"file", "bsp_sha256"}, f"stock signature {map_name}")
        artifact = _mapping(record["file"], f"stock signature {map_name} file")
        _exact_keys(artifact, {"bytes", "sha256"}, f"stock signature {map_name} file")
        _integer(artifact["bytes"], f"stock signature {map_name} bytes", 1)
        _digest(artifact["sha256"], f"stock signature {map_name} SHA-256")
        stock_bsp_hashes.append(_digest(record["bsp_sha256"], f"stock signature {map_name} BSP SHA-256"))
    if sorted(stock_bsp_hashes) != prior["source_bsp_sha256"]:
        raise B3PriorError("stock signature BSP identities differ from the stock prior")
    pairs = _list(plan["pairs"], "campaign pairs")
    if len(pairs) != 28:
        raise B3PriorError("campaign plan does not contain exactly 28 pairs")
    seeds = set()
    for ordinal, raw in enumerate(pairs):
        row = _mapping(raw, f"campaign pair {ordinal}")
        _exact_keys(row, {"ordinal", "seed", "baseline", "treatment"}, f"campaign pair {ordinal}")
        row_ordinal = _integer(row["ordinal"], f"campaign ordinal {ordinal}", 0, 27)
        if row_ordinal != ordinal:
            raise B3PriorError("campaign pair ordinals are not contiguous")
        seed = _integer(row["seed"], f"campaign seed {ordinal}", 0, 2**31 - 1)
        if seed in seeds:
            raise B3PriorError("campaign plan repeats a seed")
        seeds.add(seed)
        for lane_name, lane_bias in (
            ("baseline", None),
            ("treatment", bias),
        ):
            lane = _mapping(row[lane_name], f"campaign pair {ordinal} {lane_name}")
            _exact_keys(lane, {"map", "style", "generator_knobs"}, f"campaign pair {ordinal} {lane_name}")
            expected_map = f"{plan['campaign_id']}_{'b' if lane_name == 'baseline' else 't'}_{ordinal:02d}"
            expected_style = STYLES[ordinal // 4]
            if lane["map"] != expected_map or lane["style"] != expected_style:
                raise B3PriorError(f"campaign pair {ordinal} {lane_name} map/style was altered")
            if canonical_bytes(lane["generator_knobs"]) != canonical_bytes(_effective_knobs(expected_style, lane_bias)):
                raise B3PriorError(f"campaign pair {ordinal} {lane_name} generator knobs were altered")
    _reject_placeholder(plan, "campaign plan")
    return dict(plan)


def validate_campaign_report(value: object) -> dict[str, Any]:
    report = _mapping(value, "campaign report")
    _exact_keys(
        report,
        {
            "schema", "campaign_id", "status", "evidence_kind",
            "synthetic_claims", "community_input_used", "plan",
            "implementation", "normative_documents", "stock_prior",
            "lanes", "metrics", "diversity", "decision", "failures", "passed",
        },
        "campaign report",
    )
    if report["schema"] != CAMPAIGN_SCHEMA or report["status"] not in {"green", "red"}:
        raise B3PriorError("campaign report schema/status differs")
    if report["evidence_kind"] != "measured-compiled-atlas" or report["synthetic_claims"] is not False:
        raise B3PriorError("campaign report contains synthetic claims")
    if report["community_input_used"] is not False:
        raise B3PriorError("campaign report used community input")
    validate_implementation(report["implementation"])
    if not isinstance(report["campaign_id"], str) or not MAP_ID.fullmatch(report["campaign_id"]):
        raise B3PriorError("campaign report id is not canonical")
    plan_record = _mapping(report["plan"], "campaign plan record")
    _exact_keys(plan_record, {"bytes", "sha256"}, "campaign plan record")
    _integer(plan_record["bytes"], "campaign plan bytes", 1)
    _digest(plan_record["sha256"], "campaign plan SHA-256")
    normative = _mapping(report["normative_documents"], "campaign normative documents")
    _exact_keys(normative, {"design_sha256", "plan_sha256"}, "campaign normative documents")
    _digest(normative["design_sha256"], "campaign design SHA-256")
    _digest(normative["plan_sha256"], "campaign execution-plan SHA-256")
    stock_prior = validate_prior_pack(report["stock_prior"], "campaign stock prior")
    if stock_prior["map_count"] != 8:
        raise B3PriorError("campaign stock prior does not contain eight stock maps")
    lanes = _mapping(report["lanes"], "campaign lanes")
    _exact_keys(lanes, {"baseline", "treatment"}, "campaign lanes")
    packs: dict[str, dict[str, Any]] = {}
    for lane_name in ("baseline", "treatment"):
        lane = _mapping(lanes[lane_name], f"campaign {lane_name} lane")
        _exact_keys(
            lane,
            {"evidence", "map_count", "static_pass_count", "prior_pack", "maps_sha256"},
            f"campaign {lane_name} lane",
        )
        evidence = _mapping(lane["evidence"], f"campaign {lane_name} evidence record")
        _exact_keys(evidence, {"bytes", "sha256"}, f"campaign {lane_name} evidence record")
        _integer(evidence["bytes"], f"campaign {lane_name} evidence bytes", 1)
        _digest(evidence["sha256"], f"campaign {lane_name} evidence SHA-256")
        if lane["map_count"] != 28 or lane["static_pass_count"] != 28:
            raise B3PriorError(f"campaign {lane_name} is not a complete 28-map/static-passing lane")
        _digest(lane["maps_sha256"], f"campaign {lane_name} map-set SHA-256")
        packs[lane_name] = validate_prior_pack(lane["prior_pack"], f"campaign {lane_name} prior")
        if packs[lane_name]["map_count"] != 28:
            raise B3PriorError(f"campaign {lane_name} prior map count differs")
    metrics = _mapping(report["metrics"], "campaign metrics")
    _exact_keys(
        metrics,
        {"contract", "baseline", "treatment", "improvement_ppm", "aggregate_improvement_ppm"},
        "campaign metrics",
    )
    if canonical_bytes(metrics["contract"]) != canonical_bytes(METRIC_CONTRACT):
        raise B3PriorError("campaign report metric contract differs from frozen constants")
    distance_keys = {*METRIC_CONTRACT["primary_metrics"], "aggregate_distance_ppm"}
    for lane_name in ("baseline", "treatment"):
        distances = _mapping(metrics[lane_name], f"campaign {lane_name} distances")
        _exact_keys(distances, distance_keys, f"campaign {lane_name} distances")
        for name, distance in distances.items():
            _integer(distance, f"campaign {lane_name} {name}", 0, 1_000_000)
    improvements = _mapping(metrics["improvement_ppm"], "campaign metric improvements")
    _exact_keys(improvements, set(METRIC_CONTRACT["primary_metrics"]), "campaign metric improvements")
    for name, improvement in improvements.items():
        _signed_integer(improvement, f"campaign improvement {name}")
    _signed_integer(metrics["aggregate_improvement_ppm"], "campaign aggregate improvement")
    expected_distances = {
        lane_name: histogram_distances(stock_prior, packs[lane_name])
        for lane_name in ("baseline", "treatment")
    }
    if metrics["baseline"] != expected_distances["baseline"] or metrics["treatment"] != expected_distances["treatment"]:
        raise B3PriorError("campaign distances do not recompute from the embedded prior packs")
    expected_improvements = {
        name: expected_distances["baseline"][name] - expected_distances["treatment"][name]
        for name in METRIC_CONTRACT["primary_metrics"]
    }
    expected_aggregate = (
        expected_distances["baseline"]["aggregate_distance_ppm"]
        - expected_distances["treatment"]["aggregate_distance_ppm"]
    )
    if metrics["improvement_ppm"] != expected_improvements or metrics["aggregate_improvement_ppm"] != expected_aggregate:
        raise B3PriorError("campaign improvement arithmetic differs")
    diversity = _mapping(report["diversity"], "campaign diversity")
    _exact_keys(diversity, {"baseline", "treatment"}, "campaign diversity")
    diversity_rows: dict[str, Mapping[str, Any]] = {}
    for lane_name in ("baseline", "treatment"):
        row = _mapping(diversity[lane_name], f"campaign {lane_name} diversity")
        _exact_keys(
            row,
            {
                "map_count", "unique_layout_count", "unique_layout_ratio_ppm",
                "unique_descriptor_count", "unique_descriptor_ratio_ppm",
                "style_counts", "style_simpson_ppm",
            },
            f"campaign {lane_name} diversity",
        )
        if row["map_count"] != 28:
            raise B3PriorError(f"campaign {lane_name} diversity map count differs")
        unique_layouts = _integer(row["unique_layout_count"], f"{lane_name} unique layouts", 1, 28)
        unique_descriptors = _integer(row["unique_descriptor_count"], f"{lane_name} unique descriptors", 1, 28)
        _integer(row["unique_layout_ratio_ppm"], f"{lane_name} unique layout ratio", 0, 1_000_000)
        _integer(row["unique_descriptor_ratio_ppm"], f"{lane_name} unique descriptor ratio", 0, 1_000_000)
        _integer(row["style_simpson_ppm"], f"{lane_name} style Simpson diversity", 0, 1_000_000)
        if row["unique_layout_ratio_ppm"] != unique_layouts * 1_000_000 // 28:
            raise B3PriorError(f"campaign {lane_name} layout ratio arithmetic differs")
        if row["unique_descriptor_ratio_ppm"] != unique_descriptors * 1_000_000 // 28:
            raise B3PriorError(f"campaign {lane_name} descriptor ratio arithmetic differs")
        styles = _mapping(row["style_counts"], f"campaign {lane_name} style counts")
        if not styles or not set(styles).issubset(STYLES):
            raise B3PriorError(f"campaign {lane_name} style counts contain unknown styles")
        expanded_styles = []
        for style, count in styles.items():
            checked = _integer(count, f"campaign {lane_name} style {style}", 1, 28)
            expanded_styles.extend([style] * checked)
        if len(expanded_styles) != 28 or row["style_simpson_ppm"] != _simpson_ppm(expanded_styles):
            raise B3PriorError(f"campaign {lane_name} style-diversity arithmetic differs")
        diversity_rows[lane_name] = row
    decision = _mapping(report["decision"], "campaign decision")
    required = {
        "all_metrics_improved", "aggregate_improved", "static_pass_rate_preserved",
        "layout_diversity_passed", "style_diversity_passed",
        "descriptor_diversity_passed", "green",
    }
    _exact_keys(decision, required, "campaign decision")
    if any(not isinstance(decision[key], bool) for key in required):
        raise B3PriorError("campaign decision predicates must be booleans")
    expected_decision = {
        "all_metrics_improved": all(
            value >= METRIC_CONTRACT["minimum_improvement_ppm_each"]
            for value in expected_improvements.values()
        ),
        "aggregate_improved": expected_aggregate >= METRIC_CONTRACT["minimum_aggregate_improvement_ppm"],
        "static_pass_rate_preserved": True,
        "layout_diversity_passed": (
            diversity_rows["treatment"]["unique_layout_ratio_ppm"]
            >= METRIC_CONTRACT["required_unique_layout_ratio_ppm"]
        ),
        "style_diversity_passed": (
            diversity_rows["treatment"]["style_simpson_ppm"]
            >= METRIC_CONTRACT["minimum_style_simpson_ppm"]
            and diversity_rows["treatment"]["style_simpson_ppm"]
            >= diversity_rows["baseline"]["style_simpson_ppm"]
            - METRIC_CONTRACT["maximum_style_simpson_regression_ppm"]
        ),
        "descriptor_diversity_passed": (
            diversity_rows["treatment"]["unique_descriptor_ratio_ppm"]
            >= METRIC_CONTRACT["minimum_descriptor_unique_ratio_ppm"]
        ),
    }
    expected_decision["green"] = all(expected_decision.values())
    if dict(decision) != expected_decision:
        raise B3PriorError("campaign decision does not recompute from frozen metrics/diversity")
    if not isinstance(report["passed"], bool):
        raise B3PriorError("campaign passed flag must be a boolean")
    if report["passed"] is not decision["green"]:
        raise B3PriorError("campaign passed flag disagrees with the decision")
    expected_failures = [name for name, passed in decision.items() if name != "green" and not passed]
    if report["failures"] != expected_failures:
        raise B3PriorError("campaign failure list does not match the regression decision")
    if report["passed"] is True:
        if report["status"] != "green" or report["failures"] != []:
            raise B3PriorError("green campaign has failures or red status")
        if any(decision[key] is not True for key in required):
            raise B3PriorError("green campaign has a false gate predicate")
    _reject_placeholder(report, "campaign report")
    return dict(report)


def evaluate_campaign(
    plan_path: Path,
    stock_analysis_dir: Path,
    baseline_lane_path: Path,
    baseline_analysis_dir: Path,
    treatment_lane_path: Path,
    treatment_analysis_dir: Path,
    output_path: Path,
    *,
    repo_root: Path = ROOT,
    _implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = validate_plan(load_json(plan_path))
    plan_sha256 = file_sha256(plan_path)
    implementation = validate_implementation(
        _implementation if _implementation is not None else repository_binding(repo_root)
    )
    if plan["implementation"] != implementation:
        raise B3PriorError("current implementation identity drifted from the frozen plan")
    current_normative = {
        "design_sha256": file_sha256(repo_root / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"),
        "plan_sha256": file_sha256(repo_root / "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md"),
    }
    if plan["normative_documents"] != current_normative:
        raise B3PriorError("authoritative design/plan identity drifted from the frozen plan")
    stock_prior, stock_records = load_stock_prior(stock_analysis_dir)
    if stock_prior != plan["stock_prior"]:
        raise B3PriorError("stock prior drifted from the frozen plan")
    projected_stock = {
        name: {"file": stock_records[name]["file"], "bsp_sha256": stock_records[name]["bsp_sha256"]}
        for name in STOCK_MAPS
    }
    if projected_stock != plan["stock_signatures"]:
        raise B3PriorError("stock signature artifact identities drifted from the frozen plan")
    baseline_rows, baseline_pack = _validate_lane(
        load_json(baseline_lane_path), "baseline", plan, plan_sha256, baseline_analysis_dir,
    )
    treatment_rows, treatment_pack = _validate_lane(
        load_json(treatment_lane_path), "treatment", plan, plan_sha256, treatment_analysis_dir,
    )
    baseline_distance = histogram_distances(stock_prior, baseline_pack)
    treatment_distance = histogram_distances(stock_prior, treatment_pack)
    primary = METRIC_CONTRACT["primary_metrics"]
    improvements = {
        name: baseline_distance[name] - treatment_distance[name]
        for name in primary
    }
    all_metrics_improved = all(
        value >= METRIC_CONTRACT["minimum_improvement_ppm_each"]
        for value in improvements.values()
    )
    aggregate_improvement = (
        baseline_distance["aggregate_distance_ppm"]
        - treatment_distance["aggregate_distance_ppm"]
    )
    aggregate_improved = aggregate_improvement >= METRIC_CONTRACT["minimum_aggregate_improvement_ppm"]

    def diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        layouts = len({row["layout_sha256"] for row in rows})
        descriptors = len({row["descriptor_sha256"] for row in rows})
        styles = [str(row["style"]) for row in rows]
        return {
            "map_count": count,
            "unique_layout_count": layouts,
            "unique_layout_ratio_ppm": layouts * 1_000_000 // count,
            "unique_descriptor_count": descriptors,
            "unique_descriptor_ratio_ppm": descriptors * 1_000_000 // count,
            "style_counts": dict(sorted(Counter(styles).items())),
            "style_simpson_ppm": _simpson_ppm(styles),
        }

    baseline_diversity = diversity(baseline_rows)
    treatment_diversity = diversity(treatment_rows)
    layout_diversity = (
        treatment_diversity["unique_layout_ratio_ppm"]
        >= METRIC_CONTRACT["required_unique_layout_ratio_ppm"]
    )
    descriptor_diversity = (
        treatment_diversity["unique_descriptor_ratio_ppm"]
        >= METRIC_CONTRACT["minimum_descriptor_unique_ratio_ppm"]
    )
    style_diversity = (
        treatment_diversity["style_simpson_ppm"]
        >= METRIC_CONTRACT["minimum_style_simpson_ppm"]
        and treatment_diversity["style_simpson_ppm"]
        >= baseline_diversity["style_simpson_ppm"]
        - METRIC_CONTRACT["maximum_style_simpson_regression_ppm"]
    )
    decision = {
        "all_metrics_improved": all_metrics_improved,
        "aggregate_improved": aggregate_improved,
        "static_pass_rate_preserved": True,
        "layout_diversity_passed": layout_diversity,
        "style_diversity_passed": style_diversity,
        "descriptor_diversity_passed": descriptor_diversity,
    }
    green = all(decision.values())
    decision["green"] = green
    failures = [name for name, passed in decision.items() if name != "green" and not passed]
    report = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "status": "green" if green else "red",
        "evidence_kind": "measured-compiled-atlas",
        "synthetic_claims": False,
        "community_input_used": False,
        "plan": {"bytes": plan_path.stat().st_size, "sha256": plan_sha256},
        "implementation": implementation,
        "normative_documents": current_normative,
        "stock_prior": stock_prior,
        "lanes": {
            "baseline": {
                "evidence": {"bytes": baseline_lane_path.stat().st_size, "sha256": file_sha256(baseline_lane_path)},
                "map_count": len(baseline_rows), "static_pass_count": len(baseline_rows),
                "prior_pack": baseline_pack, "maps_sha256": sha256_bytes(canonical_bytes(baseline_rows)),
            },
            "treatment": {
                "evidence": {"bytes": treatment_lane_path.stat().st_size, "sha256": file_sha256(treatment_lane_path)},
                "map_count": len(treatment_rows), "static_pass_count": len(treatment_rows),
                "prior_pack": treatment_pack, "maps_sha256": sha256_bytes(canonical_bytes(treatment_rows)),
            },
        },
        "metrics": {
            "contract": METRIC_CONTRACT,
            "baseline": baseline_distance,
            "treatment": treatment_distance,
            "improvement_ppm": improvements,
            "aggregate_improvement_ppm": aggregate_improvement,
        },
        "diversity": {"baseline": baseline_diversity, "treatment": treatment_diversity},
        "decision": decision,
        "failures": failures,
        "passed": green,
    }
    validate_campaign_report(report)
    _exclusive_write(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze prior, seeds, treatment, and metrics")
    prepare.add_argument("--campaign-id", required=True)
    prepare.add_argument("--stock-analysis-dir", required=True, type=Path)
    prepare.add_argument("--bias", required=True, type=Path)
    prepare.add_argument("--seeds", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate", help="evaluate exact compiled-Atlas lanes")
    evaluate.add_argument("--plan", required=True, type=Path)
    evaluate.add_argument("--stock-analysis-dir", required=True, type=Path)
    evaluate.add_argument("--baseline-lane", required=True, type=Path)
    evaluate.add_argument("--baseline-analysis-dir", required=True, type=Path)
    evaluate.add_argument("--treatment-lane", required=True, type=Path)
    evaluate.add_argument("--treatment-analysis-dir", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare_campaign(
                args.campaign_id, args.stock_analysis_dir, args.bias,
                args.seeds, args.output,
            )
            summary = {"schema": PLAN_SCHEMA, "campaign_id": report["campaign_id"], "pair_count": len(report["pairs"])}
        else:
            report = evaluate_campaign(
                args.plan, args.stock_analysis_dir, args.baseline_lane,
                args.baseline_analysis_dir, args.treatment_lane,
                args.treatment_analysis_dir, args.output,
            )
            summary = {"schema": CAMPAIGN_SCHEMA, "campaign_id": report["campaign_id"], "passed": report["passed"]}
    except B3PriorError as exc:
        print(f"B3 design-prior campaign refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
