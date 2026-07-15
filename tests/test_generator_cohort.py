from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from tools import run_generator_cohort as cohort


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
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
            "hook_claim_candidates_v2": {
                "schema": "q2-hook-claim-candidates-v2",
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
        (output / f"{name}.routes.json").write_text(
            json.dumps({"map": name, "routes": [seed]}) + "\n", encoding="utf-8"
        )

    return generate


def static_pass(map_path: Path) -> dict[str, object]:
    return {"map": map_path.stem, "static_ok": True}


def test_authoritative_declaration_is_canonical_balanced_and_no_salvage() -> None:
    declaration, digest = cohort.load_declaration(DECLARATION)

    assert len(digest) == 64
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
