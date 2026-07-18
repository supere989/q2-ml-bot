from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.assemble_b2_qualification as assembler
import tools.run_b2_qualification_source as source
from tools.run_generator_cohort import (
    CONCRETE_STYLES,
    SOURCE_SUFFIXES,
    canonical_bytes,
)


SHA = "a" * 64
COMMIT = "b" * 40
TREE = "c" * 40


def _binding() -> dict[str, object]:
    return {
        "repository_commit": COMMIT,
        "repository_tree": TREE,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": SHA,
        "atlas_analyzer_authority_file_count": 7,
        "generator_sha256": "d" * 64,
        "routes_sha256": "e" * 64,
    }


def _fake_generator(
    name: str,
    seed: int,
    output: Path,
    *,
    grid: int,
    style: str,
) -> None:
    values = {
        ".map": f"map {name} {seed} {grid} {style}\n",
        ".json": "# bundle_admissible: false\n",
        ".meta.json": json.dumps({"name": name, "seed": seed, "style": style}),
        ".lattice.json": json.dumps({"map": name, "seed": seed}),
        ".routes.json": json.dumps({"map": name, "seed": seed}),
    }
    for suffix, value in values.items():
        (output / f"{name}{suffix}").write_text(value)


def _metadata(path: Path, row: dict[str, object]) -> dict[str, object]:
    value = json.loads(path.read_text())
    return {
        "name": value["name"],
        "seed": value["seed"],
        "style": value["style"],
        "grid": row["grid"],
    }


def _static(path: Path) -> dict[str, object]:
    return {"static_ok": True, "map_sha256_input": path.read_text()}


def _route(path: Path, map_id: str) -> dict[str, object]:
    value = json.loads(path.read_text())
    return {"schema": "test-route-v1", "map": map_id, "seed": value["seed"]}


def _spawn(
    path: Path, route: dict[str, object], map_id: str,
) -> dict[str, object]:
    return {
        "schema": "test-spawn-binding-v1",
        "map": map_id,
        "route_seed": route["seed"],
        "map_text": path.read_text(),
    }


def _run(
    tmp_path: Path,
    *,
    generator=_fake_generator,
    static_validator=_static,
) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    paths = {
        "source": tmp_path / "source",
        "cold": tmp_path / "cold",
        "declaration": tmp_path / "qualification-declaration.json",
        "report": tmp_path / "source-report.json",
    }
    declaration, report = source.run_source_qualification(
        qualification_id="b2q26_source_test",
        seed_base=880_000,
        source_root=paths["source"],
        cold_root=paths["cold"],
        declaration_path=paths["declaration"],
        report_path=paths["report"],
        workers=4,
        _generator=generator,
        _static_validator=static_validator,
        _metadata_validator=_metadata,
        _route_loader=_route,
        _spawn_binding=_spawn,
        _binding=_binding(),
        _retired=(set(), set(), set()),
    )
    return declaration, report, paths


def _no_retired() -> dict[str, set[object]]:
    return {
        "cohorts": set(), "declarations": set(), "maps": set(),
        "seeds": set(), "digests": set(),
    }


def test_emits_balanced_canonical_source_stage_and_cold_rebuild(
    tmp_path: Path,
) -> None:
    declaration, report, paths = _run(tmp_path)
    assert declaration["mode"] == "qualification"
    assert declaration["non_admissible"] is True
    assert declaration["retryable"] is True
    assert declaration["final_cohort_authorized"] is False
    assert len(declaration["maps"]) == 28
    assert {
        style: sum(row["style"] == style for row in declaration["maps"])
        for style in CONCRETE_STYLES
    } == {style: 4 for style in CONCRETE_STYLES}
    assert report["pass_count"] == 28
    assert report["input_report_sha256"] is None
    assert paths["declaration"].read_bytes() == canonical_bytes(declaration)
    assert paths["report"].read_bytes() == canonical_bytes(report)
    assert len(list(paths["source"].iterdir())) == 28 * len(SOURCE_SUFFIXES)
    assert len(list(paths["cold"].iterdir())) == 28 * len(SOURCE_SUFFIXES)

    retired = _no_retired()
    loaded, declaration_sha256 = assembler._validate_declaration(
        paths["declaration"], _binding(), retired
    )
    assert loaded == declaration
    summary, _, passed_maps = assembler._validate_stage_report(
        paths["report"], "source", declaration, declaration_sha256,
        _binding(), None, retired,
    )
    assert summary["pass_count"] == len(passed_maps) == 28

    for declared, row in zip(declaration["maps"], report["maps"]):
        files = {
            suffix: {
                "bytes": (paths["source"] / f"{declared['map']}{suffix}").stat().st_size,
                "sha256": source._sha256_bytes(
                    (paths["source"] / f"{declared['map']}{suffix}").read_bytes()
                ),
            }
            for suffix in SOURCE_SUFFIXES
        }
        assert row["evidence_sha256"] == source.source_evidence_sha256(
            declared["map"], files
        )


def test_static_failure_is_a_map_failure_not_an_infrastructure_claim(
    tmp_path: Path,
) -> None:
    def one_failure(path: Path) -> dict[str, object]:
        return {
            "static_ok": "open_880000" not in path.name,
            "map_sha256_input": path.read_text(),
        }

    _, report, _ = _run(tmp_path, static_validator=one_failure)
    assert report["pass_count"] == 27
    assert report["infrastructure_checks"]["source-static"] is True
    failed = [row for row in report["maps"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["criteria"]["source-static"] is False
    assert failed[0]["failures"] == [
        "source/static validation failed or differed on cold rebuild"
    ]


def test_consumer_replays_cold_source_and_rejects_forged_report(tmp_path: Path) -> None:
    declaration, report, paths = _run(tmp_path)
    replayed, raw, digest, passed = source.validate_published_qualification_source(
        declaration_path=paths["declaration"], source_root=paths["source"],
        cold_root=paths["cold"], report_path=paths["report"],
        implementation_provider=lambda _root: _binding(),
        static_validator=_static, metadata_validator=_metadata,
        route_loader=_route, spawn_binding=_spawn,
    )
    assert replayed == report
    assert digest == source._sha256_bytes(raw)
    assert len(passed) == 28

    forged = json.loads(paths["report"].read_text())
    forged["maps"][4]["evidence_sha256"] = "f" * 64
    paths["report"].write_bytes(canonical_bytes(forged))
    with pytest.raises(source.QualificationSourceError, match="source report differs"):
        source.validate_published_qualification_source(
            declaration_path=paths["declaration"], source_root=paths["source"],
            cold_root=paths["cold"], report_path=paths["report"],
            implementation_provider=lambda _root: _binding(),
            static_validator=_static, metadata_validator=_metadata,
            route_loader=_route, spawn_binding=_spawn,
        )


def test_rejects_retired_seed_before_creating_roots(tmp_path: Path) -> None:
    retired_seed = 880_000
    with pytest.raises(source.QualificationSourceError, match="retired map seed"):
        source.run_source_qualification(
            qualification_id="b2q26_source_test",
            seed_base=retired_seed,
            source_root=tmp_path / "source",
            cold_root=tmp_path / "cold",
            declaration_path=tmp_path / "declaration.json",
            report_path=tmp_path / "report.json",
            workers=2,
            _generator=_fake_generator,
            _static_validator=_static,
            _metadata_validator=_metadata,
            _route_loader=_route,
            _spawn_binding=_spawn,
            _binding=_binding(),
            _retired=(set(), set(), {retired_seed}),
        )
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "cold").exists()


def test_rejects_nondeterministic_cold_bytes_without_reports(tmp_path: Path) -> None:
    def nondeterministic(
        name: str, seed: int, output: Path, *, grid: int, style: str,
    ) -> None:
        _fake_generator(name, seed, output, grid=grid, style=style)
        if output.name == "cold" and name.endswith("open_880000"):
            (output / f"{name}.map").write_text("cold-only mutation\n")

    with pytest.raises(source.QualificationSourceError, match="changed after cold"):
        _run(tmp_path, generator=nondeterministic)
    assert not (tmp_path / "qualification-declaration.json").exists()
    assert not (tmp_path / "source-report.json").exists()


def test_real_generator_smoke_produces_source_static_map(tmp_path: Path) -> None:
    name = "b2q26_smoke_open_880000"
    source._default_generation_task(
        {"map": name, "seed": 880_000, "grid": 5, "style": "open"},
        str(tmp_path),
    )
    assert all((tmp_path / f"{name}{suffix}").is_file() for suffix in SOURCE_SUFFIXES)
    result = source._default_static_validator(tmp_path / f"{name}.map")
    assert result["static_ok"] is True
