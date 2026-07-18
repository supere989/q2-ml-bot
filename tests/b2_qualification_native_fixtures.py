"""Small canonical fixtures shared by qualification-native producer tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from tools.assemble_b2_qualification import STAGE_SCHEMA
from tools.b2_qualification_toolchain import (
    ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
)
from tools.run_generator_cohort import CONCRETE_STYLES, canonical_bytes


IMPLEMENTATION = {
    "repository_commit": "12" * 20,
    "repository_tree": "34" * 20,
    "git_clean": True,
    "atlas_analyzer_authority_sha256": "56" * 32,
    "atlas_analyzer_authority_file_count": 1,
    "generator_sha256": "78" * 32,
    "routes_sha256": "9a" * 32,
}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> Path:
    path.write_bytes(canonical_bytes(value))
    return path


def declaration() -> dict[str, Any]:
    maps = []
    for ordinal in range(28):
        style = CONCRETE_STYLES[ordinal // 4]
        maps.append({
            "ordinal": ordinal, "map": f"b2q26_fixture_{ordinal:02d}",
            "seed": 810000 + ordinal, "style": style, "grid": 5,
            "observed_heat": None,
        })
    return {
        "schema": "q2-b2-qualification-declaration-v1",
        "qualification_id": "b2q26_fixture",
        "mode": "qualification", "non_admissible": True, "retryable": True,
        "final_cohort_authorized": False,
        "generator": {"version": "v6", "grid": 5, "gym": False,
                      "observed_heat": None},
        "selection": {"required_map_count": 28,
                      "required_concrete_styles": list(CONCRETE_STYLES),
                      "required_maps_per_style": 4},
        "implementation": IMPLEMENTATION,
        "toolchain_authority_sha256": ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
        "maps": maps,
    }


def stage_report(
    stage: str, declaration_value: Mapping[str, Any], declaration_sha: str,
    input_sha: str | None, *, pass_count: int = 28,
    criteria: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    rows = []
    for row in declaration_value["maps"]:
        passed = row["ordinal"] < pass_count
        row_criteria = dict(criteria or {"fixture": True})
        if not passed:
            row_criteria[next(iter(row_criteria))] = False
        rows.append({
            "ordinal": row["ordinal"], "map": row["map"],
            "criteria": row_criteria,
            "evidence_sha256": sha(f"{stage}:{row['map']}".encode()),
            "failures": [] if passed else ["fixture lifecycle failure"],
            "passed": passed,
        })
    return {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration_value["qualification_id"],
        "mode": "qualification", "stage": stage, "non_admissible": True,
        "retryable": True, "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha,
        "implementation": IMPLEMENTATION,
        "toolchain_authority_sha256": ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
        "input_report_sha256": input_sha,
        "infrastructure_checks": {"fixture": True},
        "map_count": 28, "pass_count": pass_count, "maps": rows, "failures": [],
    }
