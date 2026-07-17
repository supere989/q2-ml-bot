from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.assemble_b2_qualification import (
    DECLARATION_SCHEMA,
    STAGE_SCHEMA,
    _validate_stage_report,
)
from tools.run_b2_qualification_compile import (
    COMPILED_SUFFIXES,
    CONCRETE_STYLES,
    _file_record,
)
from tools.run_b2_qualification_postcompile import (
    CLAIMS_SUFFIXES,
    MATERIALIZED_SUFFIXES,
    QualificationPostcompileError,
    ROOT,
    _validate_pinned_python_runtime,
    claims_qualification,
    materialize_qualification,
    stage_evidence_sha256,
    validate_published_qualification_postcompile,
)
from tools.run_generator_cohort import canonical_bytes
from tools.b2_qualification_toolchain import (
    ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
)


IMPLEMENTATION = {
    "repository_commit": "12" * 20,
    "repository_tree": "34" * 20,
    "git_clean": True,
    "atlas_analyzer_authority_sha256": "56" * 32,
    "atlas_analyzer_authority_file_count": 1,
    "generator_sha256": "78" * 32,
    "routes_sha256": "9a" * 32,
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _declaration() -> dict:
    maps = []
    for ordinal in range(28):
        style = CONCRETE_STYLES[ordinal // 4]
        maps.append({
            "ordinal": ordinal,
            "map": f"b2q26_post_{style}_{ordinal:02d}",
            "seed": 99100000 + ordinal,
            "style": style,
            "grid": 5,
            "observed_heat": None,
        })
    return {
        "schema": DECLARATION_SCHEMA,
        "qualification_id": "b2q26_postcompile_fixture",
        "mode": "qualification",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "generator": {"version": "v6", "grid": 5, "gym": False, "observed_heat": None},
        "selection": {
            "required_map_count": 28,
            "required_concrete_styles": list(CONCRETE_STYLES),
            "required_maps_per_style": 4,
        },
        "implementation": IMPLEMENTATION,
        "toolchain_authority_sha256": ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
        "maps": maps,
    }


def _prior_report(declaration: dict, stage: str, failed: set[str] | None = None) -> dict:
    failed = failed or set()
    rows = []
    for declared in declaration["maps"]:
        map_id = declared["map"]
        passed = map_id not in failed
        rows.append({
            "ordinal": declared["ordinal"],
            "map": map_id,
            "criteria": {"fixture-stage": passed},
            "evidence_sha256": _sha(f"{stage}:{map_id}".encode()),
            "failures": [] if passed else ["fixture rejection"],
            "passed": passed,
        })
    return {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": stage,
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": _sha(canonical_bytes(declaration)),
        "implementation": IMPLEMENTATION,
        "toolchain_authority_sha256": ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
        "input_report_sha256": "ab" * 32,
        "infrastructure_checks": (
            {
                "real-bsp-cm": True,
                "compiled-invariants": True,
                "fixture-infrastructure": True,
            }
            if stage == "compiled-cm-preflight"
            else {
                "authority-bound": True,
                "materialized-membership": True,
                "fixture-infrastructure": True,
            }
        ),
        "map_count": 28,
        "pass_count": 28 - len(failed),
        "maps": rows,
        "failures": [],
    }


def _fixture(tmp_path: Path, *, prior_failed: set[str] | None = None) -> dict:
    declaration = _declaration()
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_bytes(canonical_bytes(declaration))
    compile_path = tmp_path / "compile-report.json"
    compile_path.write_bytes(canonical_bytes({"fixture": "compile"}))
    prior = _prior_report(declaration, "compiled-cm-preflight", prior_failed)
    prior["input_report_sha256"] = _sha(compile_path.read_bytes())
    prior_path = tmp_path / "preflight-report.json"
    prior_path.write_bytes(canonical_bytes(prior))
    evidence = tmp_path / "compiled-cm-evidence"
    evidence.mkdir()
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    for declared in declaration["maps"]:
        for suffix in COMPILED_SUFFIXES:
            (compiled / f"{declared['map']}{suffix}").write_text(
                f"{declared['map']}:{suffix}\n", encoding="ascii"
            )
    authority_paths = {}
    for name in ("cm", "pmove", "hook", "fall", "hook_attestation"):
        path = tmp_path / name
        path.write_text(name, encoding="ascii")
        authority_paths[name] = path
    return {
        "declaration": declaration,
        "declaration_path": declaration_path,
        "prior": prior_path,
        "compile_report": compile_path,
        "prior_evidence": evidence,
        "python": tmp_path / "python",
        "compiled": compiled,
        "materialize_staging": tmp_path / "materialize-staging",
        "materialized": tmp_path / "materialized",
        "logs": tmp_path / "logs",
        "materialize_report": tmp_path / "materialize-report.json",
        "claims_staging": tmp_path / "claims-staging",
        "claims": tmp_path / "claims",
        "claims_report": tmp_path / "claims-report.json",
        **authority_paths,
    }


def _authorities(**kwargs) -> dict:
    return {
        name: _file_record(kwargs[f"{name}_oracle"])
        for name in ("cm", "pmove", "hook", "fall")
    } | {
        "hook_attestation": _file_record(kwargs["hook_attestation"]),
        "b1_gate": {"bytes": 1, "sha256": "bc" * 32},
    }


def _materializer(**kwargs) -> None:
    map_id = kwargs["map_id"]
    stage = kwargs["stage_root"]
    (stage / f"{map_id}.json").write_text(
        f"sealed-runtime:{map_id}\n", encoding="ascii"
    )
    (stage / f"{map_id}.hook-materialization.json").write_bytes(
        canonical_bytes({"map": map_id, "passed": True, "schema": "fixture-v4"})
    )


def _compiled_cm_validator(**kwargs):
    declaration = json.loads(Path(kwargs["declaration_path"]).read_text())
    report, raw = json.loads(Path(kwargs["report_path"]).read_text()), Path(
        kwargs["report_path"]
    ).read_bytes()
    assert report["input_report_sha256"] == _sha(
        Path(kwargs["compile_report_path"]).read_bytes()
    )
    passed = {row["map"] for row in report["maps"] if row["passed"]}
    return report, raw, _sha(raw), passed


def _runtime(_python: Path, _root: Path) -> dict:
    return {"schema": "fixture-pinned-runtime-v1"}


def _run_materialize(paths: dict, *, materializer=_materializer) -> dict:
    return materialize_qualification(
        declaration_path=paths["declaration_path"],
        prior_report_path=paths["prior"],
        prior_evidence_root=paths["prior_evidence"],
        compile_report_path=paths["compile_report"],
        compiled_root=paths["compiled"],
        staging_root=paths["materialize_staging"],
        materialized_root=paths["materialized"],
        log_root=paths["logs"],
        report_path=paths["materialize_report"],
        cm_oracle=paths["cm"],
        pmove_oracle=paths["pmove"],
        hook_oracle=paths["hook"],
        fall_oracle=paths["fall"],
        hook_attestation=paths["hook_attestation"],
        python_runtime=paths["python"],
        implementation_provider=lambda _root: dict(IMPLEMENTATION),
        authority_provider=_authorities,
        map_materializer=materializer,
        compiled_cm_validator=_compiled_cm_validator,
        runtime_provider=_runtime,
    )


def _claim_builder(path: Path) -> dict:
    return {"schema": "fixture-claims-v1", "map": path.stem, "passed": True}


def _run_claims(paths: dict, *, builder=_claim_builder) -> dict:
    return claims_qualification(
        declaration_path=paths["declaration_path"],
        prior_report_path=paths["materialize_report"],
        upstream_report_path=paths["prior"],
        materialized_root=paths["materialized"],
        staging_root=paths["claims_staging"],
        claims_root=paths["claims"],
        report_path=paths["claims_report"],
        implementation_provider=lambda _root: dict(IMPLEMENTATION),
        claim_builder=builder,
    )


def test_materialization_and_claims_publish_exact_chained_stages(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    materialized = _run_materialize(paths)
    assert materialized["stage"] == "materialization"
    assert materialized["pass_count"] == 28
    assert len(list(paths["materialized"].iterdir())) == 196
    claims = _run_claims(paths)
    assert claims["stage"] == "claims"
    assert claims["pass_count"] == 28
    assert len(list(paths["claims"].iterdir())) == 224
    assert claims["input_report_sha256"] == _sha(
        paths["materialize_report"].read_bytes()
    )
    records = {
        suffix: _file_record(paths["claims"] / f"{claims['maps'][0]['map']}{suffix}")
        for suffix in CLAIMS_SUFFIXES
    }
    assert claims["maps"][0]["evidence_sha256"] == stage_evidence_sha256(
        claims["maps"][0]["map"], True, records
    )
    for report_path, stage, prior_path in (
        (paths["materialize_report"], "materialization", paths["prior"]),
        (paths["claims_report"], "claims", paths["materialize_report"]),
    ):
        summary, _digest, passed = _validate_stage_report(
            report_path, stage, paths["declaration"],
            _sha(paths["declaration_path"].read_bytes()), IMPLEMENTATION,
            _sha(prior_path.read_bytes()), {"digests": set()},
        )
        assert summary["pass_count"] == 28
        assert len(passed) == 28


def test_prior_failure_propagates_without_breaking_exact_membership(tmp_path: Path) -> None:
    declaration = _declaration()
    rejected = {declaration["maps"][5]["map"]}
    paths = _fixture(tmp_path, prior_failed=rejected)
    materialized = _run_materialize(paths)
    assert materialized["pass_count"] == 27
    assert len(list(paths["materialized"].iterdir())) == 27 * len(MATERIALIZED_SUFFIXES)
    claims = _run_claims(paths)
    assert claims["pass_count"] == 27
    rejected_row = next(row for row in claims["maps"] if row["map"] in rejected)
    assert rejected_row["criteria"]["prior-stage-passed"] is False
    assert rejected_row["passed"] is False
    assert len(list(paths["claims"].iterdir())) == 27 * len(CLAIMS_SUFFIXES)


def test_one_materializer_failure_publishes_honest_sparse_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    failed = paths["declaration"]["maps"][3]["map"]

    def failing(**kwargs) -> None:
        if kwargs["map_id"] == failed:
            raise RuntimeError("fixture materializer failure")
        _materializer(**kwargs)

    report = _run_materialize(paths, materializer=failing)
    assert report["pass_count"] == 27
    assert report["failures"] == []
    assert len(list(paths["materialized"].iterdir())) == 27 * len(MATERIALIZED_SUFFIXES)
    row = next(item for item in report["maps"] if item["map"] == failed)
    assert row["passed"] is False
    assert "fixture materializer failure" in row["failures"][0]


def test_one_claim_failure_publishes_honest_sparse_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _run_materialize(paths)
    failed = paths["declaration"]["maps"][7]["map"]

    def failing(path: Path) -> dict:
        if path.stem == failed:
            raise ValueError("fixture claims failure")
        return _claim_builder(path)

    report = _run_claims(paths, builder=failing)
    assert report["pass_count"] == 27
    assert report["failures"] == []
    assert len(list(paths["claims"].iterdir())) == 27 * len(CLAIMS_SUFFIXES)


def test_authority_drift_fails_materialization_before_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    calls = 0

    def drifting(**kwargs) -> dict:
        nonlocal calls
        calls += 1
        result = _authorities(**kwargs)
        if calls > 1:
            result["cm"] = {**result["cm"], "sha256": "ff" * 32}
        return result

    with pytest.raises(QualificationPostcompileError, match="input stability"):
        materialize_qualification(
            declaration_path=paths["declaration_path"],
            prior_report_path=paths["prior"],
            prior_evidence_root=paths["prior_evidence"],
            compile_report_path=paths["compile_report"],
            compiled_root=paths["compiled"],
            staging_root=paths["materialize_staging"],
            materialized_root=paths["materialized"], log_root=paths["logs"],
            report_path=paths["materialize_report"], cm_oracle=paths["cm"],
            pmove_oracle=paths["pmove"], hook_oracle=paths["hook"],
            fall_oracle=paths["fall"], hook_attestation=paths["hook_attestation"],
            python_runtime=paths["python"],
            implementation_provider=lambda _root: dict(IMPLEMENTATION),
            authority_provider=drifting, map_materializer=_materializer,
            compiled_cm_validator=_compiled_cm_validator,
            runtime_provider=_runtime,
        )
    assert not paths["materialized"].exists()


def test_reports_and_publications_are_exclusive(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    materialized = _run_materialize(paths)
    assert paths["materialize_report"].read_bytes() == canonical_bytes(materialized)
    with pytest.raises(QualificationPostcompileError, match="output must be fresh"):
        _run_materialize(paths)
    claims = _run_claims(paths)
    assert paths["claims_report"].read_bytes() == canonical_bytes(claims)
    with pytest.raises(QualificationPostcompileError, match="output must be fresh"):
        _run_claims(paths)


def test_twenty_of_twenty_eight_propagates_through_sparse_claims(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    rejected = {
        row["map"] for row in paths["declaration"]["maps"][:8]
    }

    def selective(**kwargs) -> None:
        if kwargs["map_id"] in rejected:
            raise RuntimeError("semantic materialization rejection")
        _materializer(**kwargs)

    materialized = _run_materialize(paths, materializer=selective)
    claims = _run_claims(paths)
    assert materialized["pass_count"] == claims["pass_count"] == 20
    assert len(list(paths["materialized"].iterdir())) == 20 * len(MATERIALIZED_SUFFIXES)
    assert len(list(paths["claims"].iterdir())) == 20 * len(CLAIMS_SUFFIXES)


def test_consumer_replays_sparse_postcompile_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    declaration = _declaration()
    rejected = {row["map"] for row in declaration["maps"][:8]}
    paths = _fixture(tmp_path, prior_failed=rejected)
    materialized = _run_materialize(paths)
    claims = _run_claims(paths)
    for row in materialized["maps"]:
        if not row["passed"]:
            continue
        map_id = row["map"]
        (paths["logs"] / f"{map_id}.stdout.json").write_bytes(b"{}\n")
        (paths["logs"] / f"{map_id}.stderr.log").write_bytes(b"")

    kwargs = {
        "declaration_path": paths["declaration_path"],
        "compile_report_path": paths["compile_report"],
        "compiled_root": paths["compiled"],
        "compiled_cm_report_path": paths["prior"],
        "compiled_cm_evidence_root": paths["prior_evidence"],
        "materialization_report_path": paths["materialize_report"],
        "materialized_root": paths["materialized"],
        "materialization_log_root": paths["logs"],
        "claims_report_path": paths["claims_report"],
        "claims_root": paths["claims"],
        "cm_oracle": paths["cm"],
        "pmove_oracle": paths["pmove"],
        "hook_oracle": paths["hook"],
        "fall_oracle": paths["fall"],
        "hook_attestation": paths["hook_attestation"],
        "python_runtime": paths["python"],
        "implementation_provider": lambda _root: dict(IMPLEMENTATION),
        "authority_provider": _authorities,
        "runtime_provider": _runtime,
        "result_validator": lambda *args, **kwargs: None,
        "claim_builder": _claim_builder,
        "compiled_cm_validator": _compiled_cm_validator,
    }
    replay = validate_published_qualification_postcompile(**kwargs)
    assert len(replay["compiled_cm_passed"]) == 20
    assert len(replay["materialization_passed"]) == 20
    assert len(replay["claims_passed"]) == 20

    original_material_report = paths["materialize_report"].read_bytes()
    forged_material_report = json.loads(original_material_report)
    passing_row = next(row for row in forged_material_report["maps"] if row["passed"])
    passing_row["evidence_sha256"] = "00" * 32
    paths["materialize_report"].write_bytes(canonical_bytes(forged_material_report))
    with pytest.raises(QualificationPostcompileError):
        validate_published_qualification_postcompile(**kwargs)
    paths["materialize_report"].write_bytes(original_material_report)

    passing_map = next(row["map"] for row in claims["maps"] if row["passed"])
    (paths["claims"] / f"{passing_map}.generator-claims.json").write_bytes(
        canonical_bytes({"schema": "forged-claims-v1", "map": passing_map})
    )
    with pytest.raises(QualificationPostcompileError):
        validate_published_qualification_postcompile(**kwargs)


def test_claims_reject_forged_raw_upstream_chain_before_staging(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _run_materialize(paths)
    forged = json.loads(paths["materialize_report"].read_text())
    forged["input_report_sha256"] = "00" * 32
    paths["materialize_report"].write_bytes(canonical_bytes(forged))
    with pytest.raises(QualificationPostcompileError, match="hash"):
        _run_claims(paths)
    assert not paths["claims_staging"].exists()
    assert not paths["claims"].exists()


def test_wrong_python_path_is_rejected_before_any_preflight(tmp_path: Path) -> None:
    wrong = tmp_path / "python"
    wrong.write_bytes(b"not the pinned runtime\n")
    wrong.chmod(0o755)
    with pytest.raises(QualificationPostcompileError, match="pinned runtime"):
        _validate_pinned_python_runtime(wrong, ROOT)
