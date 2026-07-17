from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.assemble_b2_qualification import (
    DECLARATION_SCHEMA,
    STAGE_SCHEMA,
    _validate_stage_report,
)
from tools.run_b2_qualification_compile import COMPILED_SUFFIXES
from tools.run_b2_qualification_compiled_cm import (
    QualificationCompiledCmError,
    _expected_physics_identity,
    run_qualification_compiled_cm,
    validate_published_qualification_compiled_cm,
)
from tools.run_compiled_cm_preflight import (
    ESCAPE_DISTANCE_UNITS,
    ESCAPE_STEP_UNITS,
    MILLIUNITS,
    SPAWN_LINK_LIFT,
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
TOOL_IDENTITY = "ab" * 32
SOURCE_CLOSURE = "cd" * 32


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": _sha256(payload)}


def _declaration() -> dict[str, object]:
    maps = []
    for ordinal in range(28):
        style = CONCRETE_STYLES[ordinal // 4]
        maps.append({
            "ordinal": ordinal,
            "map": f"b2q26_cm_{style}_{ordinal:02d}",
            "seed": 91_000_000 + ordinal,
            "style": style,
            "grid": 5,
            "observed_heat": None,
        })
    return {
        "schema": DECLARATION_SCHEMA,
        "qualification_id": "b2q26_compiled_cm_fixture",
        "mode": "qualification",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "generator": {
            "version": "v6", "grid": 5, "gym": False,
            "observed_heat": None,
        },
        "selection": {
            "required_map_count": 28,
            "required_concrete_styles": list(CONCRETE_STYLES),
            "required_maps_per_style": 4,
        },
        "implementation": IMPLEMENTATION,
        "maps": maps,
    }


def _compile_report(
    declaration: dict[str, object], declaration_sha256: str,
) -> dict[str, object]:
    rows = []
    for row in declaration["maps"]:
        rows.append({
            "ordinal": row["ordinal"],
            "map": row["map"],
            "criteria": {
                "source-stage-bound": True,
                "q2tool-exit-zero": True,
                "q2tool-not-timed-out": True,
                "ibsp38-lightdata": True,
                "compiled-stage-published": True,
            },
            "evidence_sha256": _sha256(
                f"compile:{row['map']}".encode("ascii")
            ),
            "failures": [],
            "passed": True,
        })
    return {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": "compile",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": IMPLEMENTATION,
        "input_report_sha256": "ef" * 32,
        "infrastructure_checks": {
            "real-q2tool": True,
            "compiled-membership": True,
            "bounded-parallel-workers": True,
            "input-stability": True,
            "ibsp38-lightdata": True,
            "exclusive-publication": True,
        },
        "map_count": 28,
        "pass_count": 28,
        "maps": rows,
        "failures": [],
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    declaration = _declaration()
    declaration_path = tmp_path / "qualification-declaration.json"
    declaration_path.write_bytes(canonical_bytes(declaration))
    compile_report = _compile_report(
        declaration, _sha256(declaration_path.read_bytes())
    )
    compile_path = tmp_path / "compile-report.json"
    compile_path.write_bytes(canonical_bytes(compile_report))
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    for row in declaration["maps"]:
        for suffix in COMPILED_SUFFIXES:
            (compiled / f"{row['map']}{suffix}").write_bytes(
                f"{row['map']}:{suffix}\n".encode("ascii")
            )
    oracle = tmp_path / "q2-cm-oracle"
    oracle.write_bytes(b"qualification CM oracle\n")
    oracle.chmod(0o755)
    gate = SimpleNamespace(
        cm_executable_sha256=_record(oracle)["sha256"],
        oracle_tool_identity=TOOL_IDENTITY,
        oracle_source_closure_sha256=SOURCE_CLOSURE,
    )
    return {
        "declaration": declaration,
        "declaration_path": declaration_path,
        "compile_report": compile_report,
        "compile_path": compile_path,
        "compiled": compiled,
        "oracle": oracle,
        "evidence": tmp_path / "compiled-cm-evidence",
        "report": tmp_path / "compiled-cm-report.json",
        "gate": gate,
    }


def _spawn(ordinal: int) -> dict[str, object]:
    return {
        "entity_ordinal": ordinal,
        "authored_origin_milliunits": [ordinal * 512_000, 0, 24_000],
        "engine_link_lift_milliunits": round(SPAWN_LINK_LIFT * MILLIUNITS),
        "standing_clear": True,
        "crouched_clear": True,
        "supported": True,
        "support_drop_milliunits": 9_000,
        "column_clearance_milliunits": 100_000,
        "column_clear_96": True,
        "basic_escape": {
            "distance_milliunits": ESCAPE_DISTANCE_UNITS * MILLIUNITS,
            "support_step_milliunits": ESCAPE_STEP_UNITS * MILLIUNITS,
            "passing_direction_indices": [0],
            "passed": True,
        },
        "failures": [],
        "passed": True,
    }


def _validator(
    failed: set[int] | None = None,
    forged: set[int] | None = None,
    mutate: tuple[int, Path] | None = None,
):
    failed = failed or set()
    forged = forged or set()

    def validate(row, compiled, cm_oracle, limits, oracle_sha, tool, closure):
        del limits
        assert tool == TOOL_IDENTITY
        assert closure == SOURCE_CLOSURE
        ordinal = row["ordinal"]
        bsp_path = compiled / f"{row['map']}.bsp"
        bsp = _record(bsp_path)
        origins = [[index * 512_000, 0, 24_000] for index in range(8)]
        failures = [] if ordinal not in failed else ["synthetic CM rejection"]
        spawns = [_spawn(index) for index in range(8)]
        if ordinal in forged:
            spawns[0]["column_clearance_milliunits"] = 92_000
        result = {
            "ordinal": ordinal,
            "map": row["map"],
            "bsp": bsp,
            "compiled_lightdata": {
                "bytes": 3, "sha256": _sha256(b"lit"), "present": True,
            },
            "source_spawn_origins_milliunits": origins,
            "compiled_spawn_origins_milliunits": origins,
            "spawn_origin_sets_match": True,
            "minimum_spawn_xy_separation_milliunits": 512_000,
            "oracle": {
                "executable_sha256": oracle_sha,
                "tool_identity": tool,
                "physics_identity": _expected_physics_identity(
                    tool, bsp["sha256"]
                ),
                "map_sha256": bsp["sha256"],
            },
            "spawn_count": 8,
            "spawns": spawns,
            "basic_hazard_containment": {
                "declared_hazard_count": 0,
                "checked_hazard_count": 0,
                "hazards": [],
                "failures": [],
                "passed": True,
            },
            "failures": failures,
            "passed": not failures,
        }
        if mutate is not None and ordinal == mutate[0]:
            mutate[1].write_bytes(b"changed during CM preflight\n")
        return result

    return validate


def _run(
    paths: dict[str, object], *, validator=None,
    gate=None,
) -> dict[str, object]:
    return run_qualification_compiled_cm(
        declaration_path=paths["declaration_path"],
        compile_report_path=paths["compile_path"],
        compiled_root=paths["compiled"],
        cm_oracle=paths["oracle"],
        evidence_root=paths["evidence"],
        report_path=paths["report"],
        jobs=8,
        oracle_batch_timeout_seconds=1.0,
        implementation_provider=lambda _root: dict(IMPLEMENTATION),
        gate_loader=lambda _root: gate or paths["gate"],
        preflight_implementation_provider=lambda _root: {
            "schema": "fixture-preflight-implementation-v1",
            "source_closure_sha256": "fe" * 32,
        },
        map_validator=validator or _validator(),
    )


def _replay(paths: dict[str, object]):
    return validate_published_qualification_compiled_cm(
        declaration_path=paths["declaration_path"],
        compile_report_path=paths["compile_path"],
        compiled_root=paths["compiled"], cm_oracle=paths["oracle"],
        evidence_root=paths["evidence"], report_path=paths["report"],
        implementation_provider=lambda _root: dict(IMPLEMENTATION),
        gate_loader=lambda _root: paths["gate"],
        preflight_implementation_provider=lambda _root: {
            "schema": "fixture-preflight-implementation-v1",
            "source_closure_sha256": "fe" * 32,
        },
    )


def test_emits_all_real_map_results_and_preserves_retryable_failures(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    report = _run(paths, validator=_validator(failed={3, 19}))
    assert report["stage"] == "compiled-cm-preflight"
    assert report["non_admissible"] is True
    assert report["retryable"] is True
    assert report["final_cohort_authorized"] is False
    assert report["map_count"] == 28
    assert report["pass_count"] == 26
    assert [row["ordinal"] for row in report["maps"]] == list(range(28))
    assert [row["ordinal"] for row in report["maps"] if not row["passed"]] == [3, 19]
    assert paths["report"].read_bytes() == canonical_bytes(report)
    assert len(list(paths["evidence"].iterdir())) == 28
    for row, evidence_path in zip(
        report["maps"], sorted(paths["evidence"].iterdir())
    ):
        evidence = json.loads(evidence_path.read_text())
        assert row["evidence_sha256"] == _sha256(evidence_path.read_bytes())
        assert evidence["real_preflight_result"]["map"] == row["map"]
        assert evidence["compile_report_sha256"] == _sha256(
            paths["compile_path"].read_bytes()
        )

    summary, digest, passed = _validate_stage_report(
        paths["report"],
        "compiled-cm-preflight",
        paths["declaration"],
        _sha256(paths["declaration_path"].read_bytes()),
        IMPLEMENTATION,
        _sha256(paths["compile_path"].read_bytes()),
        {"digests": set()},
    )
    assert summary["pass_count"] == len(passed) == 26
    assert digest == _sha256(paths["report"].read_bytes())


def test_forged_pass_cannot_bypass_independent_compiled_invariants(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    report = _run(paths, validator=_validator(forged={7}))
    row = report["maps"][7]
    assert report["pass_count"] == 27
    assert row["criteria"]["real-bsp-cm"] is True
    assert row["criteria"]["compiled-invariants"] is False
    assert row["passed"] is False
    assert row["failures"] == [
        "real compiled-CM result is incomplete or internally inconsistent"
    ]


def test_compiled_input_drift_prevents_any_evidence_publication(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    target = paths["compiled"] / (
        f"{paths['declaration']['maps'][10]['map']}.routes.json"
    )
    with pytest.raises(QualificationCompiledCmError, match="input changed"):
        _run(paths, validator=_validator(mutate=(10, target)))
    assert not paths["evidence"].exists()
    assert not paths["report"].exists()


def test_stale_b1_cm_identity_fails_before_running_maps(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    stale = SimpleNamespace(
        cm_executable_sha256="00" * 32,
        oracle_tool_identity=TOOL_IDENTITY,
        oracle_source_closure_sha256=SOURCE_CLOSURE,
    )
    with pytest.raises(QualificationCompiledCmError, match="fresh B1"):
        _run(paths, gate=stale)
    assert not paths["evidence"].exists()
    assert not paths["report"].exists()


def test_partial_or_final_compile_population_is_not_silently_wrapped(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    compile_report = dict(paths["compile_report"])
    compile_report["mode"] = "final"
    paths["compile_path"].write_bytes(canonical_bytes(compile_report))
    with pytest.raises(QualificationCompiledCmError, match="compile report rejected"):
        _run(paths)
    assert not paths["evidence"].exists()
    assert not paths["report"].exists()


def test_consumer_replays_every_raw_evidence_document(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _run(paths)
    report, raw, digest, passed = _replay(paths)
    assert report["pass_count"] == len(passed) == 28
    assert digest == _sha256(raw)


@pytest.mark.parametrize("mutation", ("evidence", "extra", "compiled", "compile"))
def test_consumer_rejects_forged_or_drifted_predecessor(
    tmp_path: Path, mutation: str,
) -> None:
    paths = _fixture(tmp_path)
    _run(paths)
    if mutation == "evidence":
        target = sorted(paths["evidence"].iterdir())[4]
        value = json.loads(target.read_text())
        value["passed"] = False
        target.write_bytes(canonical_bytes(value))
    elif mutation == "extra":
        (paths["evidence"] / "extra.evidence.json").write_bytes(b"{}\n")
    elif mutation == "compiled":
        map_id = paths["declaration"]["maps"][6]["map"]
        (paths["compiled"] / f"{map_id}.bsp").write_bytes(b"drift\n")
    else:
        value = json.loads(paths["compile_path"].read_text())
        value["pass_count"] = 27
        paths["compile_path"].write_bytes(canonical_bytes(value))
    with pytest.raises(QualificationCompiledCmError):
        _replay(paths)
