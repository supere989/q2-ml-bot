from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from tools import compile_generated_cohort as compiler
from tools import materialize_generated_cohort as materializer
from tools import retired_cohort_registry as registry
from tools import run_generated_atlas_campaign as atlas_campaign
from tools import run_generator_claim_campaign as claim_campaign
from tools import run_generator_cohort as cohort


ROOT = Path(__file__).resolve().parents[1]
CURRENT_ALIAS = ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
NAMED_71438 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71438-DECLARATION.json"
)
NAMED_71439 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71439-DECLARATION.json"
)
NAMED_71440 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71440-DECLARATION.json"
)
NAMED_71441 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71441-DECLARATION.json"
)
NAMED_71442 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71442-DECLARATION.json"
)
NAMED_71443 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71443-DECLARATION.json"
)
FAILURE_71443 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71443-FAILURE.json"
)
RETIRED_DECLARATIONS = (
    NAMED_71438,
    NAMED_71439,
    NAMED_71440,
    NAMED_71441,
    NAMED_71442,
    NAMED_71443,
)


def _write_fresh_declaration(path: Path) -> Path:
    declaration, _sha256 = cohort.load_declaration(NAMED_71438)
    declaration["cohort_id"] = "b2g26_test_fresh_99000"
    for row in declaration["maps"]:
        row["map"] = f"b2g26_test_fresh_{row['ordinal']:02d}"
        row["seed"] = 99000000 + row["ordinal"]
    path.write_bytes(cohort.canonical_bytes(declaration))
    return path


def _write_failed_authority(
    root: Path,
    number: int,
    *,
    duplicate_map: str | None = None,
) -> str:
    directory = root / "docs/multires"
    directory.mkdir(parents=True, exist_ok=True)
    declaration, _sha256 = cohort.load_declaration(NAMED_71438)
    declaration["cohort_id"] = f"b2g26_final_{number}"
    for row in declaration["maps"]:
        row["map"] = f"retired_{number}_{row['ordinal']:02d}"
        row["seed"] = number * 1000 + row["ordinal"]
    if duplicate_map is not None:
        declaration["maps"][0]["map"] = duplicate_map
    relative = f"docs/multires/B2-GENERATED-COHORT-{number}-DECLARATION.json"
    declaration_path = root / relative
    declaration_path.write_bytes(cohort.canonical_bytes(declaration))
    _loaded, declaration_sha256 = cohort.load_declaration(declaration_path)
    authority = {
        "admission": {
            "older_population_reuse_allowed": False,
            "passing_subset_allowed": False,
            "permanently_non_admissible": True,
            "regeneration_under_same_declaration_allowed": False,
            "replacement_declaration_status": "pending-test",
            "replacement_member_allowed": False,
            "retry_under_same_declaration_allowed": False,
            "salvage_allowed": False,
        },
        "cohort_id": declaration["cohort_id"],
        "declaration": {"path": relative, "sha256": declaration_sha256},
        "evidence": {},
        "failure": {},
        "schema": registry.FAILURE_SCHEMA,
        "status": "permanently-failed-test",
    }
    failure_path = (
        directory / f"B2-GENERATED-COHORT-{number}-FAILURE.json"
    )
    failure_path.write_bytes(cohort.canonical_bytes(authority))
    return declaration["maps"][0]["map"]


@pytest.mark.parametrize("declaration_path", RETIRED_DECLARATIONS)
def test_retired_cohort_cannot_generate_without_creating_outputs(
    tmp_path: Path, declaration_path: Path,
) -> None:
    root = tmp_path / "generated"

    with pytest.raises(
        registry.RetiredCohortRegistryError, match="permanently retired"
    ):
        cohort.generate_source_freeze(
            declaration_path,
            root / "source",
            root / "cold",
            root / "source-freeze.json",
            _binding={},
        )

    assert not root.exists()


@pytest.mark.parametrize("declaration_path", RETIRED_DECLARATIONS)
def test_retired_cohort_cannot_compile_without_creating_outputs(
    tmp_path: Path, declaration_path: Path,
) -> None:
    root = tmp_path / "compiled"

    with pytest.raises(
        compiler.CompileCohortError, match="permanently retired"
    ) as refusal:
        compiler.compile_generated_cohort(
            declaration_path,
            root / "source",
            root / "staging",
            root / "published",
            root / "logs",
            root / "reports/compile.json",
            root / "q2tool",
            root / "basedir",
        )

    assert isinstance(refusal.value.__cause__, registry.RetiredCohortRegistryError)
    assert not root.exists()


@pytest.mark.parametrize("declaration_path", RETIRED_DECLARATIONS)
def test_retired_cohort_cannot_materialize_without_creating_outputs(
    tmp_path: Path, declaration_path: Path,
) -> None:
    root = tmp_path / "materialized"

    with pytest.raises(
        registry.RetiredCohortRegistryError, match="permanently retired"
    ):
        materializer.materialize_cohort(
            declaration_path=declaration_path,
            compiled_dir=root / "compiled",
            stage_dir=root / "staging",
            materialized_dir=root / "published",
            log_dir=root / "logs",
            report_path=root / "reports/materialize.json",
            cm_oracle=root / "cm",
            pmove_oracle=root / "pmove",
            hook_oracle=root / "hook",
            fall_oracle=root / "fall",
            hook_attestation=root / "hook-attestation.json",
        )

    assert not root.exists()


@pytest.mark.parametrize("declaration_path", RETIRED_DECLARATIONS)
def test_retired_cohort_cannot_prepare_claims_or_build_atlas(
    tmp_path: Path, declaration_path: Path,
) -> None:
    claims_root = tmp_path / "claims-campaign"
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="permanently retired"
    ):
        claim_campaign.prepare_claims(
            declaration_path,
            claims_root / "materialized",
            claims_root / "claims",
        )
    assert not claims_root.exists()

    atlas_root = tmp_path / "atlas-campaign"
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="permanently retired"
    ):
        atlas_campaign.build_atlas_campaign(
            declaration_path,
            atlas_root / "claims",
            atlas_root / "analysis",
            atlas_root / "diagnostics",
            atlas_root / "reports/build.json",
        )
    assert not atlas_root.exists()


def test_genuinely_fresh_declaration_is_admitted(tmp_path: Path) -> None:
    declaration_path = _write_fresh_declaration(tmp_path / "fresh.json")
    declaration, declaration_sha256 = cohort.load_declaration(declaration_path)

    assert (
        registry.require_unretired_declaration(
            declaration_path, declaration, declaration_sha256
        )
        is None
    )


@pytest.mark.parametrize("declaration_path", [NAMED_71440])
def test_named_71440_is_retired(declaration_path: Path) -> None:
    declaration, declaration_sha256 = cohort.load_declaration(declaration_path)

    assert declaration["cohort_id"] == "b2g26_final_71440"
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71440.*permanently retired"
    ):
        registry.require_unretired_declaration(
            declaration_path, declaration, declaration_sha256
        )


def test_named_71439_remains_retired() -> None:
    declaration, declaration_sha256 = cohort.load_declaration(NAMED_71439)
    assert declaration["cohort_id"] == "b2g26_final_71439"
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71439.*permanently retired"
    ):
        registry.require_unretired_declaration(
            NAMED_71439, declaration, declaration_sha256
        )


@pytest.mark.parametrize("declaration_path", [NAMED_71441])
def test_alias_and_named_71441_are_retired(declaration_path: Path) -> None:
    current, current_sha256 = cohort.load_declaration(declaration_path)

    assert current["cohort_id"] == "b2g26_final_71441"
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71441.*permanently retired"
    ):
        registry.require_unretired_declaration(
            declaration_path, current, current_sha256
        )


def test_current_alias_71443_is_retired_until_replaced() -> None:
    current, current_sha256 = cohort.load_declaration(CURRENT_ALIAS)

    assert current["cohort_id"] == "b2g26_final_71443"
    assert CURRENT_ALIAS.read_bytes() == NAMED_71443.read_bytes()
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71443.*permanently retired"
    ):
        registry.require_unretired_declaration(
            CURRENT_ALIAS, current, current_sha256
        )


def test_71443_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71443.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "da89be636079b0cc38583281113002f0578d2608c5a31af052fca8c03d05f723"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == "permanently-failed-test-runtime-preflight"
    assert authority["cohort_id"] == "b2g26_final_71443"
    assert authority["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71443-DECLARATION.json",
        "sha256": "d890e151cbc3446622a8c0f5fdd2bd23352583c6431e1484262587c3c7246713",
    }
    assert authority["admission"]["permanently_non_admissible"] is True
    assert authority["admission"]["source_stage_published"] is True
    assert authority["admission"]["compiled_stage_published"] is True
    for stage in ("materialized", "claims", "analysis"):
        assert authority["admission"][f"{stage}_stage_published"] is False
    for control in (
        "dyn_execution_allowed",
        "older_population_reuse_allowed",
        "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed",
        "retry_under_same_declaration_allowed",
        "salvage_allowed",
        "substitution_allowed",
    ):
        assert authority["admission"][control] is False

    evidence = authority["evidence"]
    assert evidence["implementation"]["repository_commit"] == (
        "dedb8392a6fc96946a8f9e346791c89ecdcc7e14"
    )
    assert evidence["implementation"]["repository_tree"] == (
        "0a3a66d7c7159bab66cd411c7597e3e49223afdd"
    )
    assert evidence["source_freeze"]["report_sha256"] == (
        "6e748dd45bfd013cfd9c57f2ec60289b9abf40da946e511d771efe096d02a456"
    )
    assert evidence["wsl_compile"]["report_sha256"] == (
        "c0c7f8c857e8ef60f0f74b959fef6b34f458fc69223146d7245ce2e79de76d84"
    )
    assert evidence["test_runtime_failure"] == {
        "commands_launched": 2,
        "diagnostic_log": {
            "bytes": 5243,
            "path": "/home/raymond/q2-multires-isolated/B2/b2g26_final_71443-failure/pytest-diagnostic.log",
            "sha256": "196d25d0de40e4333dda9fe4c946e84ae571133554cb72e5ffa1c835bef1bb2d",
        },
        "output_root_exists": False,
        "pytest_exit_code": 2,
        "pytest_version": "9.1.1",
        "raw_logs_published": False,
        "report_published": False,
        "runner_error": "pytest log lacks one unambiguous pass summary",
        "test_interpreter": {
            "path": "/usr/bin/python3.10",
            "sha256": "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86",
            "version": "3.10.12",
        },
        "tests_executed": 0,
        "zstandard_import_available": False,
    }
    assert authority["failure"]["phase"] == (
        "out-of-order-test-runtime-preflight"
    )
    assert authority["failure"]["publication_contract"] == {
        "analysis_run": False,
        "claims_prepare_run": False,
        "compiled_cm_preflight_run": False,
        "compiled_maps_passed": 28,
        "compiled_stage_published": True,
        "dyn_run": False,
        "gate_run": False,
        "maps_declared": 28,
        "materialization_run": False,
        "source_maps_passed": 28,
        "source_stage_published": True,
        "test_campaign_run": True,
        "test_report_published": False,
        "training_run": False,
    }


@pytest.mark.parametrize(
    ("identity", "pattern"),
    [("map", "map ID .* permanently retired"), ("seed", "seed .* permanently retired")],
)
def test_fresh_cohort_cannot_reuse_retired_map_or_seed(
    tmp_path: Path, identity: str, pattern: str,
) -> None:
    declaration_path = _write_fresh_declaration(tmp_path / "fresh.json")
    declaration, _sha256 = cohort.load_declaration(declaration_path)
    retired, _retired_sha256 = cohort.load_declaration(NAMED_71443)
    declaration["maps"][0][identity] = retired["maps"][0][identity]
    declaration_path.write_bytes(cohort.canonical_bytes(declaration))
    declaration, declaration_sha256 = cohort.load_declaration(declaration_path)

    with pytest.raises(registry.RetiredCohortRegistryError, match=pattern):
        registry.require_unretired_declaration(
            declaration_path, declaration, declaration_sha256
        )


def test_malformed_failure_registry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_dir = tmp_path / "docs/multires"
    registry_dir.mkdir(parents=True)
    (registry_dir / "B2-GENERATED-COHORT-bad-FAILURE.json").write_bytes(b"{}\n")
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "FAILURE_REGISTRY_DIR", registry_dir)
    declaration_path = _write_fresh_declaration(tmp_path / "fresh.json")
    declaration, declaration_sha256 = cohort.load_declaration(declaration_path)

    with pytest.raises(
        registry.RetiredCohortRegistryError, match="filename is malformed"
    ):
        registry.require_unretired_declaration(
            declaration_path, declaration, declaration_sha256
        )


def test_contradictory_failure_registry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_map = _write_failed_authority(tmp_path, 99001)
    _write_failed_authority(tmp_path, 99002, duplicate_map=duplicate_map)
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        registry, "FAILURE_REGISTRY_DIR", tmp_path / "docs/multires"
    )
    declaration_path = _write_fresh_declaration(tmp_path / "fresh.json")
    declaration, declaration_sha256 = cohort.load_declaration(declaration_path)

    with pytest.raises(
        registry.RetiredCohortRegistryError,
        match="contradictory retired map ID",
    ):
        registry.require_unretired_declaration(
            declaration_path, declaration, declaration_sha256
        )


@pytest.mark.parametrize(
    ("declaration_path", "cohort_id", "declaration_sha256"),
    [
        (
            NAMED_71438,
            "b2g26_final_71438",
            "bebe7c2c63711c399d34780f3297a622f9d28d1c9751511473ec1ed4815a58c2",
        ),
        (
            NAMED_71439,
            "b2g26_final_71439",
            "374b1052ea4a15404dfd52ebf831f9d5eccda488ea5a51d3d41d0e83ee083811",
        ),
        (
            NAMED_71440,
            "b2g26_final_71440",
            "d71b86a109bb359f927457d3904cef3116d83c59104cc85b3a87dd43ddc791b2",
        ),
        (
            NAMED_71441,
            "b2g26_final_71441",
            "5929532e0edae77b48073abccf4a4f3afdbacfb6905d1eadfb7f18d1dc5ba151",
        ),
        (
            NAMED_71443,
            "b2g26_final_71443",
            "d890e151cbc3446622a8c0f5fdd2bd23352583c6431e1484262587c3c7246713",
        ),
    ],
)
def test_retired_declaration_remains_available_for_read_only_forensics(
    tmp_path: Path,
    declaration_path: Path,
    cohort_id: str,
    declaration_sha256: str,
) -> None:
    declaration, actual_sha256 = cohort.load_declaration(declaration_path)

    membership = cohort.verify_stage_membership(
        declaration, tmp_path / "absent", "source"
    )

    assert declaration["cohort_id"] == cohort_id
    assert actual_sha256 == declaration_sha256
    assert membership["passed"] is False
    assert membership["expected_map_count"] == 28


def test_generate_cli_reports_retirement_without_traceback_or_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "generate-cli"
    monkeypatch.setattr(sys, "argv", [
        "run_generator_cohort.py",
        "generate",
        "--declaration",
        str(NAMED_71441),
        "--output-dir",
        str(root / "source"),
        "--cold-dir",
        str(root / "cold"),
        "--report",
        str(root / "report.json"),
    ])

    assert cohort.main() == 1

    captured = capsys.readouterr()
    assert "generator cohort failed:" in captured.err
    assert "permanently retired" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_materialize_cli_reports_retirement_without_traceback_or_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "materialize-cli"
    arguments = [
        "--declaration", str(NAMED_71441),
        "--compiled-dir", str(root / "compiled"),
        "--stage-dir", str(root / "stage"),
        "--materialized-dir", str(root / "materialized"),
        "--log-dir", str(root / "logs"),
        "--report", str(root / "report.json"),
        "--cm-oracle", str(root / "cm"),
        "--pmove-oracle", str(root / "pmove"),
        "--hook-oracle", str(root / "hook"),
        "--fall-oracle", str(root / "fall"),
        "--hook-parity-attestation", str(root / "attestation.json"),
    ]

    assert materializer.main(arguments) == 1

    captured = capsys.readouterr()
    assert "generated cohort materialization failed:" in captured.err
    assert "permanently retired" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_claim_prepare_cli_reports_retirement_without_traceback_or_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "claims-cli"

    assert claim_campaign.main([
        "prepare",
        "--declaration", str(NAMED_71441),
        "--materialized-dir", str(root / "materialized"),
        "--claims-dir", str(root / "claims"),
        "--output", str(root / "report.json"),
    ]) == 1

    captured = capsys.readouterr()
    assert "generator claim campaign failed:" in captured.err
    assert "permanently retired" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_atlas_build_cli_reports_retirement_without_traceback_or_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "atlas-cli"

    assert atlas_campaign.main([
        "--declaration", str(NAMED_71441),
        "--claims-dir", str(root / "claims"),
        "--analysis-dir", str(root / "analysis"),
        "--diagnostics-dir", str(root / "diagnostics"),
        "--output", str(root / "report.json"),
    ]) == 1

    captured = capsys.readouterr()
    assert "generated Atlas campaign failed:" in captured.err
    assert "permanently retired" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()
