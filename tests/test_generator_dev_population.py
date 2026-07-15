from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import pytest

from tools import audit_generator_dev_population as audit
from tools import run_generator_cohort as cohort


ROOT = Path(__file__).resolve().parents[1]
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
    defect: str | None = None,
    defective_map: str | None = None,
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
        map_bytes = f"layout:{name}:{seed}:{style}:{grid}\n".encode()
        if defect == "cold_drift" and name == defective_map and output.name == "cold":
            map_bytes += b"cold-drift\n"
        (output / f"{name}.map").write_bytes(map_bytes)
        (output / f"{name}.json").write_text(
            "# development hook projection\n", encoding="utf-8"
        )
        (output / f"{name}.meta.json").write_text(
            json.dumps({
                "name": name,
                "seed": seed,
                "style": style,
                "generator": "v6",
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / f"{name}.lattice.json").write_text(
            json.dumps({"name": name, "seed": seed}) + "\n",
            encoding="utf-8",
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
            }
            for item_id in range(4)
        ]
        nodes.append({
            "id": 4,
            "type": "spawn",
            "x": base + 100,
            "y": 100,
            "z": 24,
            "room": 0,
        })
        routes = [
            {"archetype": "offense", "start_room": 0, "node_ids": [0, 1]},
            {"archetype": "survival", "start_room": 0, "node_ids": [1, 2]},
            {"archetype": "control", "start_room": 0, "node_ids": [2, 3]},
            {"archetype": "balanced", "start_room": 0, "node_ids": [3, 0]},
        ]
        if name == defective_map:
            if defect == "duplicate_item_origin":
                nodes[1]["x"], nodes[1]["y"], nodes[1]["z"] = (
                    nodes[0]["x"], nodes[0]["y"], nodes[0]["z"]
                )
            elif defect == "duplicate_endpoint":
                routes[0]["node_ids"] = [0, 0]
            elif defect == "zero_length_leg":
                nodes[4]["x"], nodes[4]["y"], nodes[4]["z"] = (
                    nodes[0]["x"], nodes[0]["y"], nodes[0]["z"]
                )
            elif defect == "missing_archetype":
                routes[3]["archetype"] = "control"
            elif defect == "unassigned_spawn":
                nodes[4]["room"] = -1
            elif defect == "unassigned_item":
                nodes[0]["room"] = -1
            elif defect == "item_spawn_collision":
                nodes.append({
                    "id": 5,
                    "type": "item",
                    "class": "item_collision",
                    "x": nodes[4]["x"],
                    "y": nodes[4]["y"],
                    "z": nodes[4]["z"],
                    "room": 0,
                })
            elif defect == "one_endpoint":
                routes[0]["node_ids"] = [0]
            elif defect == "non_item_endpoint":
                routes[0]["node_ids"] = [4, 1]
        by_id = {node["id"]: node for node in nodes}
        for route in routes:
            loop = [
                nodes[4],
                *(by_id[node_id] for node_id in route["node_ids"]),
                nodes[4],
            ]
            route["dist"] = round(sum(
                math.dist(
                    (source["x"], source["y"], source["z"]),
                    (target["x"], target["y"], target["z"]),
                )
                for source, target in zip(loop, loop[1:])
            ))
        if name == defective_map and defect == "distance_mismatch":
            routes[0]["dist"] += 2
        (output / f"{name}.routes.json").write_text(
            json.dumps({"version": 1, "nodes": nodes, "routes": routes}) + "\n",
            encoding="utf-8",
        )

    return generate


def static_pass(map_path: Path) -> dict[str, object]:
    return {"map": map_path.stem, "static_ok": True}


def test_fixed_matrix_is_balanced_and_disjoint_from_final_declaration() -> None:
    rows = audit.development_matrix()
    final_ids, final_seeds, _cohort_id, _digest = audit._final_reservations(
        audit.FINAL_DECLARATION
    )

    assert len(rows) == 56
    assert [row["ordinal"] for row in rows] == list(range(56))
    assert Counter(row["style"] for row in rows) == {
        style: 8 for style in cohort.CONCRETE_STYLES
    }
    assert {row["seed"] for row in rows} == {
        71_425_000 + style_index * 100 + offset
        for style_index in range(7)
        for offset in range(8)
    }
    assert {row["map"] for row in rows}.isdisjoint(final_ids)
    assert {row["seed"] for row in rows}.isdisjoint(final_seeds)


def test_complete_double_generation_publishes_canonical_development_report(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    report_path = tmp_path / "reports/development-audit.json"

    report = audit.audit_development_population(
        tmp_path / "primary",
        tmp_path / "cold",
        report_path,
        _generator=fake_generator_factory(calls),
        _static_validator=static_pass,
        _binding=binding(),
    )

    assert len(calls) == 112
    assert Counter(call["directory"] for call in calls) == {
        "primary": 56,
        "cold": 56,
    }
    assert report["passed"] is True
    assert report["map_count"] == report["required_map_count"] == 56
    assert report["style_counts"] == {
        style: 8 for style in sorted(cohort.CONCRETE_STYLES)
    }
    assert report["unique_layout_count"] == 56
    assert report["source_static_pass_count"] == 56
    assert report["metadata_identity_pass_count"] == 56
    assert report["route_count"] == 224
    assert report["all_route_archetypes_exactly_once"] is True
    assert report["globally_unique_item_origins"] is True
    assert report["all_item_nodes_floor_assigned"] is True
    assert report["item_spawn_origin_collisions"] == 0
    assert report["minimum_distinct_item_endpoints_per_route"] == 2
    assert report["all_route_endpoints_are_items"] is True
    assert report["published_dist_matches_endpoint_loop"] is True
    assert report["duplicate_route_endpoints"] == 0
    assert report["zero_length_route_legs"] == 0
    assert report["all_spawns_and_route_endpoints_floor_assigned"] is True
    assert report["exact_source_file_count_per_directory"] == 280
    assert report["cold_rebuild"] == {
        "distinct_empty_directories": True,
        "all_five_suffixes_byte_identical": True,
    }
    assert report["evidence_scope"] == "development-only"
    assert report["final_cohort_admissible"] is False
    assert report["bundle_admissible"] is False
    assert report["atlas_admissible"] is False
    assert report["compile_performed"] is False
    assert report["deployment_performed"] is False
    assert report["final_cohort_exclusion"]["development_ids_disjoint"] is True
    assert report["final_cohort_exclusion"]["development_seeds_disjoint"] is True
    assert report_path.read_bytes() == cohort.canonical_bytes(report)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("duplicate_item_origin", "item origins are not globally unique"),
        ("duplicate_endpoint", "duplicate or invalid endpoints"),
        ("zero_length_leg", "zero-length route legs"),
        ("missing_archetype", "each route archetype exactly once"),
        ("unassigned_spawn", "spawn node 4 is not floor-assigned"),
        ("unassigned_item", "item node 0 is not floor-assigned"),
        ("item_spawn_collision", "item node 5 overlaps spawn node 4"),
        ("one_endpoint", "requires at least two item endpoints"),
        ("non_item_endpoint", "endpoint node 4 is not an item"),
        ("distance_mismatch", "differs from endpoint-loop geometry"),
    ],
)
def test_route_defect_rejects_population_without_report(
    tmp_path: Path, defect: str, message: str
) -> None:
    bad_map = audit.development_matrix()[0]["map"]
    report_path = tmp_path / "development-audit.json"

    with pytest.raises(audit.DevelopmentPopulationError, match=message):
        audit.audit_development_population(
            tmp_path / "primary",
            tmp_path / "cold",
            report_path,
            _generator=fake_generator_factory(
                [], defect=defect, defective_map=bad_map
            ),
            _static_validator=static_pass,
            _binding=binding(),
        )

    assert not report_path.exists()


def test_cold_byte_drift_rejects_population_without_report(tmp_path: Path) -> None:
    bad_map = audit.development_matrix()[9]["map"]
    report_path = tmp_path / "development-audit.json"

    with pytest.raises(
        audit.DevelopmentPopulationError,
        match=rf"{bad_map}\.map differs across fresh generations",
    ):
        audit.audit_development_population(
            tmp_path / "primary",
            tmp_path / "cold",
            report_path,
            _generator=fake_generator_factory(
                [], defect="cold_drift", defective_map=bad_map
            ),
            _static_validator=static_pass,
            _binding=binding(),
        )

    assert not report_path.exists()


def test_final_id_or_seed_is_explicitly_rejected_before_generation(
    tmp_path: Path,
) -> None:
    final_declaration, _digest = cohort.load_declaration(audit.FINAL_DECLARATION)
    final_row = final_declaration["maps"][0]
    matrix = audit.development_matrix()
    matrix[0]["map"] = final_row["map"]

    with pytest.raises(
        audit.DevelopmentPopulationError, match="rejects final cohort ID"
    ):
        audit.audit_development_population(
            tmp_path / "id-primary",
            tmp_path / "id-cold",
            tmp_path / "id-report.json",
            _matrix=matrix,
            _binding=binding(),
        )

    matrix = audit.development_matrix()
    matrix[0]["seed"] = final_row["seed"]
    with pytest.raises(
        audit.DevelopmentPopulationError, match="rejects final cohort seed"
    ):
        audit.audit_development_population(
            tmp_path / "seed-primary",
            tmp_path / "seed-cold",
            tmp_path / "seed-report.json",
            _matrix=matrix,
            _binding=binding(),
        )

    assert not (tmp_path / "id-report.json").exists()
    assert not (tmp_path / "seed-report.json").exists()
