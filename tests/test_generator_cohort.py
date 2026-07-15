from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import pytest

from tools import run_generator_cohort as cohort


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
DECLARATION_71433 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71433-DECLARATION.json"
)
HEX64 = "a" * 64
GIT40 = "b" * 40


def binding() -> dict[str, object]:
    return {
        "repository_commit": GIT40,
        "repository_tree": "c" * 40,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": HEX64,
        "atlas_analyzer_authority_file_count": 19,
        "generator_sha256": "d" * 64,
        "routes_sha256": "e" * 64,
    }


def fake_generator_factory(
    calls: list[dict[str, object]],
    *,
    cold_mismatch: str | None = None,
    cold_route_mismatch: str | None = None,
    invalid_route: str | None = None,
    wrong_meta_seed: str | None = None,
):
    def generate(
        name: str,
        seed: int,
        output: Path,
        *,
        grid: int,
        style: str,
    ) -> None:
        calls.append({
            "name": name,
            "seed": seed,
            "directory": output.name,
            "grid": grid,
            "style": style,
        })
        layout = f"map:{name}:seed:{seed}:style:{style}:grid:{grid}\n".encode()
        if output.name == "cold" and name == cold_mismatch:
            layout += b"cold-drift\n"
        (output / f"{name}.map").write_bytes(layout)
        (output / f"{name}.json").write_text(
            "# raw hook projection\n# bundle_admissible: false\n",
            encoding="utf-8",
        )
        meta = {
            "name": name,
            "seed": seed + (1 if name == wrong_meta_seed else 0),
            "style": style,
            "generator": "v6",
            "hook_claim_candidates_v4": {
                "schema": "q2-hook-claim-candidates-v4",
                "status": "unproven",
                "bundle_admissible": False,
                "records": [{"claim": index} for index in range(6)],
            },
        }
        (output / f"{name}.meta.json").write_text(
            json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / f"{name}.lattice.json").write_text(
            json.dumps({"map": name, "seed": seed}) + "\n", encoding="utf-8"
        )
        base = seed * 10
        nodes = [
            {
                "id": item_id,
                "type": "item",
                "class": f"item_{item_id}",
                "x": base + item_id * 10,
                "y": item_id * 20,
                "z": 24,
                "room": 0,
                "source_component": 0,
            }
            for item_id in range(4)
        ]
        nodes.append({
            "id": 4, "type": "spawn", "x": base + 100,
            "y": 100, "z": 24, "room": 0, "source_component": 0,
        })
        if name == invalid_route:
            nodes[1]["x"], nodes[1]["y"], nodes[1]["z"] = (
                nodes[0]["x"], nodes[0]["y"], nodes[0]["z"]
            )
        if output.name == "cold" and name == cold_route_mismatch:
            nodes[0]["x"] += 1
        routes = [
            {"archetype": "offense", "start_room": 0, "start_node_id": 4, "source_component": 0, "node_ids": [0, 1]},
            {"archetype": "survival", "start_room": 0, "start_node_id": 4, "source_component": 0, "node_ids": [1, 2]},
            {"archetype": "control", "start_room": 0, "start_node_id": 4, "source_component": 0, "node_ids": [2, 3]},
            {"archetype": "balanced", "start_room": 0, "start_node_id": 4, "source_component": 0, "node_ids": [3, 0]},
        ]
        by_id = {node["id"]: node for node in nodes}
        for route in routes:
            loop = [
                nodes[4], *(by_id[node_id] for node_id in route["node_ids"]),
                nodes[4],
            ]
            route["dist"] = round(sum(
                math.dist(
                    (source["x"], source["y"], source["z"]),
                    (target["x"], target["y"], target["z"]),
                )
                for source, target in zip(loop, loop[1:])
            ))
        (output / f"{name}.routes.json").write_text(
            json.dumps({
                "version": 2, "nodes": nodes, "edges": [], "routes": routes,
            }) + "\n",
            encoding="utf-8",
        )

    return generate


def static_pass(map_path: Path) -> dict[str, object]:
    return {"map": map_path.stem, "static_ok": True}


def test_authoritative_declaration_is_canonical_balanced_and_no_salvage() -> None:
    declaration, digest = cohort.load_declaration(DECLARATION)
    assert DECLARATION.read_bytes() == DECLARATION_71433.read_bytes()
    style_bases = (
        ("open", 71433000),
        ("towers", 71433100),
        ("canyon", 71433200),
        ("pits", 71433300),
        ("arena_open", 71433400),
        ("arena_vertical", 71433500),
        ("arena_lanes", 71433600),
    )
    expected = [
        {
            "grid": 5,
            "map": f"b2g26_{style}_{base + offset}",
            "observed_heat": None,
            "ordinal": style_index * 4 + offset,
            "seed": base + offset,
            "style": style,
        }
        for style_index, (style, base) in enumerate(style_bases)
        for offset in range(4)
    ]

    assert len(digest) == 64
    assert declaration["cohort_id"] == "b2g26_final_71433"
    assert declaration["maps"] == expected
    assert declaration["selection"] == {
        "timing": "declared-before-generation",
        "policy": "all-or-nothing",
        "replacement_allowed": False,
        "salvage_allowed": False,
        "required_map_count": 28,
        "required_concrete_styles": list(cohort.CONCRETE_STYLES),
        "required_maps_per_style": 4,
    }
    assert Counter(row["style"] for row in declaration["maps"]) == {
        style: 4 for style in cohort.CONCRETE_STYLES
    }
    assert len({row["seed"] for row in declaration["maps"]}) == 28
    assert len({row["map"] for row in declaration["maps"]}) == 28
    assert all(row["grid"] == 5 for row in declaration["maps"])
    assert all(row["observed_heat"] is None for row in declaration["maps"])


def test_generate_publishes_only_a_complete_double_built_source_freeze(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    primary = tmp_path / "primary"
    cold = tmp_path / "cold"
    report_path = tmp_path / "reports/source-freeze.json"

    report = cohort.generate_source_freeze(
        DECLARATION,
        primary,
        cold,
        report_path,
        _generator=fake_generator_factory(calls),
        _static_validator=static_pass,
        _binding=binding(),
    )

    assert len(calls) == 56
    assert Counter(call["directory"] for call in calls) == {
        "primary": 28,
        "cold": 28,
    }
    assert all(call["grid"] == 5 for call in calls)
    assert Counter(call["style"] for call in calls[:28]) == {
        style: 4 for style in cohort.CONCRETE_STYLES
    }
    assert report["passed"] is True
    assert report["map_count"] == 28
    assert report["unique_layout_count"] == 28
    assert report["route_contract_pass_count"] == 28
    assert all(
        row["route_contract"][
            "all_selected_endpoints_share_source_standing_component"
        ] is True
        for row in report["maps"]
    )
    assert report["style_counts"] == {
        style: 4 for style in sorted(cohort.CONCRETE_STYLES)
    }
    assert report["implementation"] == binding()
    assert report["cold_rebuild"] == {
        "fresh_process_required": False,
        "independent_directory": True,
        "file_count": 140,
        "all_file_bytes_match": True,
    }
    assert report_path.read_bytes() == cohort.canonical_bytes(report)
    assert cohort.verify_stage_membership(
        cohort.load_declaration(DECLARATION)[0], primary, "source"
    )["passed"] is True

    with pytest.raises(cohort.GeneratorCohortError, match="already exists"):
        cohort.generate_source_freeze(
            DECLARATION,
            tmp_path / "new-primary",
            tmp_path / "new-cold",
            report_path,
            _generator=fake_generator_factory([]),
            _static_validator=static_pass,
            _binding=binding(),
        )


def test_cold_byte_drift_rejects_whole_cohort_without_report(tmp_path: Path) -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    bad_map = declaration["maps"][7]["map"]
    report_path = tmp_path / "source-freeze.json"

    with pytest.raises(
        cohort.GeneratorCohortError,
        match=rf"{bad_map}\.map differs across fresh generations",
    ):
        cohort.generate_source_freeze(
            DECLARATION,
            tmp_path / "primary",
            tmp_path / "cold",
            report_path,
            _generator=fake_generator_factory([], cold_mismatch=bad_map),
            _static_validator=static_pass,
            _binding=binding(),
        )

    assert not report_path.exists()


def test_invalid_source_route_contract_rejects_whole_cohort(tmp_path: Path) -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    bad_map = declaration["maps"][3]["map"]
    report_path = tmp_path / "source-freeze.json"

    with pytest.raises(
        cohort.GeneratorCohortError,
        match=rf"{bad_map} item origins are not globally unique",
    ):
        cohort.generate_source_freeze(
            DECLARATION,
            tmp_path / "primary",
            tmp_path / "cold",
            report_path,
            _generator=fake_generator_factory([], invalid_route=bad_map),
            _static_validator=static_pass,
            _binding=binding(),
        )

    assert not report_path.exists()


def test_cold_route_report_mismatch_rejects_whole_cohort(tmp_path: Path) -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    bad_map = declaration["maps"][11]["map"]
    report_path = tmp_path / "source-freeze.json"

    with pytest.raises(
        cohort.GeneratorCohortError,
        match=rf"{bad_map} cold route contract differs",
    ):
        cohort.generate_source_freeze(
            DECLARATION,
            tmp_path / "primary",
            tmp_path / "cold",
            report_path,
            _generator=fake_generator_factory([], cold_route_mismatch=bad_map),
            _static_validator=static_pass,
            _binding=binding(),
        )

    assert not report_path.exists()


def test_metadata_identity_drift_rejects_whole_cohort(tmp_path: Path) -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    bad_map = declaration["maps"][13]["map"]

    with pytest.raises(
        cohort.GeneratorCohortError,
        match=rf"{bad_map} metadata seed differs from declaration",
    ):
        cohort.generate_source_freeze(
            DECLARATION,
            tmp_path / "primary",
            tmp_path / "cold",
            tmp_path / "source-freeze.json",
            _generator=fake_generator_factory([], wrong_meta_seed=bad_map),
            _static_validator=static_pass,
            _binding=binding(),
        )


def test_exact_membership_rejects_same_count_replacement_and_extra_files(
    tmp_path: Path,
) -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    stage = tmp_path / "stage"
    stage.mkdir()
    fake_generator_factory([])(
        declaration["maps"][0]["map"],
        declaration["maps"][0]["seed"],
        stage,
        grid=5,
        style=declaration["maps"][0]["style"],
    )
    for row in declaration["maps"][1:]:
        fake_generator_factory([])(
            row["map"], row["seed"], stage, grid=5, style=row["style"]
        )
    good = cohort.verify_stage_membership(declaration, stage, "source")
    assert good["passed"] is True

    missing = f"{declaration['maps'][0]['map']}.map"
    (stage / missing).unlink()
    replacement = "b2g26_replacement_99999999.map"
    (stage / replacement).write_bytes(b"replacement\n")
    bad = cohort.verify_stage_membership(declaration, stage, "source")
    assert bad["passed"] is False
    assert bad["actual_file_count"] == bad["expected_file_count"]
    assert f"missing file {missing}" in bad["failures"]
    assert f"unexpected file {replacement}" in bad["failures"]

    (stage / "unscoped-summary.json").write_text("{}\n", encoding="utf-8")
    bad_extra = cohort.verify_stage_membership(declaration, stage, "source")
    assert "unexpected file unscoped-summary.json" in bad_extra["failures"]


def test_analysis_membership_is_a_separate_exact_stage(tmp_path: Path) -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    for row in declaration["maps"]:
        for suffix in cohort.STAGE_SUFFIXES["analysis"]:
            (analysis / f"{row['map']}{suffix}").write_bytes(
                f"{row['map']}:{suffix}\n".encode()
            )

    report = cohort.verify_stage_membership(declaration, analysis, "analysis")
    assert report["passed"] is True
    assert report["actual_file_count"] == 28 * 8
    assert not (analysis / f"{declaration['maps'][0]['map']}.map").exists()


def test_repository_binding_refuses_dirty_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments[0] == "status":
            return "?? untracked-evidence"
        return GIT40

    monkeypatch.setattr(cohort, "_git_output", fake_git)
    with pytest.raises(cohort.GeneratorCohortError, match="repository is not clean"):
        cohort.repository_binding(ROOT)


def test_declaration_mutation_cannot_enable_salvage() -> None:
    declaration, _ = cohort.load_declaration(DECLARATION)
    declaration["selection"]["salvage_allowed"] = True

    with pytest.raises(cohort.GeneratorCohortError, match="salvage_allowed"):
        cohort.validate_declaration(declaration)
