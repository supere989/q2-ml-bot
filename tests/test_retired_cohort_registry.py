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
NAMED_71444 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71444-DECLARATION.json"
)
NAMED_71445 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71445-DECLARATION.json"
)
NAMED_71446 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71446-DECLARATION.json"
)
NAMED_71447 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71447-DECLARATION.json"
)
NAMED_71448 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71448-DECLARATION.json"
)
NAMED_71449 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71449-DECLARATION.json"
)
NAMED_71450 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71450-DECLARATION.json"
)
NAMED_71451 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71451-DECLARATION.json"
)
NAMED_71452 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71452-DECLARATION.json"
)
FAILURE_71443 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71443-FAILURE.json"
)
FAILURE_71444 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71444-FAILURE.json"
)
FAILURE_71445 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71445-FAILURE.json"
)
FAILURE_71446 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71446-FAILURE.json"
)
FAILURE_71447 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71447-FAILURE.json"
)
FAILURE_71448 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71448-FAILURE.json"
)
FAILURE_71449 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71449-FAILURE.json"
)
FAILURE_71450 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71450-FAILURE.json"
)
FAILURE_71451 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71451-FAILURE.json"
)
FAILURE_71452 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71452-FAILURE.json"
)
RETIRED_DECLARATIONS = (
    NAMED_71438,
    NAMED_71439,
    NAMED_71440,
    NAMED_71441,
    NAMED_71442,
    NAMED_71443,
    NAMED_71444,
    NAMED_71445,
    NAMED_71446,
    NAMED_71447,
    NAMED_71448,
    NAMED_71449,
    NAMED_71450,
    NAMED_71451,
    NAMED_71452,
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


@pytest.mark.parametrize("declaration_path", [NAMED_71447])
def test_named_71447_is_retired(declaration_path: Path) -> None:
    current, current_sha256 = cohort.load_declaration(declaration_path)

    assert current["cohort_id"] == "b2g26_final_71447"
    assert current_sha256 == (
        "76c0ffc41ff80cb4b9f0ea6648240a73b55f0a7933970f8f2e2fd05a086cb4aa"
    )
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71447.*permanently retired"
    ):
        registry.require_unretired_declaration(
            declaration_path, current, current_sha256
        )


def test_named_71448_is_retired() -> None:
    current, current_sha256 = cohort.load_declaration(NAMED_71448)

    assert current["cohort_id"] == "b2g26_final_71448"
    assert current_sha256 == (
        "0b48462a8cd8dfb752a73b711954616dd22d45d857748d316505bd17c976262a"
    )
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71448.*permanently retired"
    ):
        registry.require_unretired_declaration(
            NAMED_71448, current, current_sha256
        )


def test_named_71449_is_retired() -> None:
    current, current_sha256 = cohort.load_declaration(NAMED_71449)

    assert current["cohort_id"] == "b2g26_final_71449"
    assert current_sha256 == (
        "7d36a6a634b81db0c293dff3e7daa5c3dfa284f931a2a4202187c56a75f2f5f6"
    )
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71449.*permanently retired"
    ):
        registry.require_unretired_declaration(
            NAMED_71449, current, current_sha256
        )


def test_named_71450_is_retired() -> None:
    current, current_sha256 = cohort.load_declaration(NAMED_71450)

    assert current["cohort_id"] == "b2g26_final_71450"
    assert current_sha256 == (
        "d02c7c0737cf38be314394dd30e0293dcdf0b80c004efc1e0d072abc72f437c4"
    )
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71450.*permanently retired"
    ):
        registry.require_unretired_declaration(
            NAMED_71450, current, current_sha256
        )


@pytest.mark.parametrize("declaration_path", [CURRENT_ALIAS, NAMED_71452])
def test_current_alias_and_named_71452_are_retired(
    declaration_path: Path,
) -> None:
    retired, retired_sha256 = cohort.load_declaration(declaration_path)

    assert retired["cohort_id"] == "b2g26_final_71452"
    assert CURRENT_ALIAS.read_bytes() == NAMED_71452.read_bytes()
    assert retired_sha256 == (
        "eb9d761d5cc48c3b2ad7dbca3ee9e232884fffc241c20aea76ed363893f0baaf"
    )
    with pytest.raises(
        registry.RetiredCohortRegistryError, match="71452.*permanently retired"
    ):
        registry.require_unretired_declaration(
            declaration_path, retired, retired_sha256
        )


def test_71452_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71452.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "951fc1184f5eb21db5415a0d6d88f896e311865dfc6b5c38ae21d0203ae4fb5d"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-dyn-origin-binding-atlas-manifest-canonicality"
    )
    assert authority["cohort_id"] == "b2g26_final_71452"
    assert authority["declaration"]["sha256"] == (
        "eb9d761d5cc48c3b2ad7dbca3ee9e232884fffc241c20aea76ed363893f0baaf"
    )
    assert authority["failure"]["phase"] == (
        "dyn-origin-binding-atlas-manifest-canonicality"
    )
    dyn = authority["evidence"]["dyn"]
    assert dyn["phase_a"]["passed"] is True
    assert dyn["origin_binding"]["attempt_count"] == 1
    assert dyn["origin_binding"]["exit_code"] == 1
    assert dyn["origin_binding"]["report_published"] is False
    assert dyn["origin_binding"]["output_directory_exists"] is False
    assert dyn["promoted_representative"]["atlas_manifest_writer_canonical"] is True
    assert dyn["promoted_representative"]["atlas_manifest_sorted_canonical"] is False
    contract = authority["failure"]["publication_contract"]
    assert contract["source_stage_published"] is True
    assert contract["analysis_stage_published"] is True
    assert contract["dyn_run"] is True
    assert contract["dyn_passed"] is False
    assert contract["gate_run"] is False
    assert contract["training_run"] is False
    assert authority["evidence"]["source_authorization"]["consumed"] is True
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False
    assert authority["admission"]["replacement_declaration_status"] == (
        "pending-atlas-binder-fix-and-fresh-qualification"
    )


def test_71451_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71451.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "83d11b7bafd6a669c67fd8ae4cf7b8e990d9e30bf0a448fb3424b604d34551d2"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-dyn-expected-origin-mismatch"
    )
    assert authority["cohort_id"] == "b2g26_final_71451"
    assert authority["failure"]["phase"] == "dyn-artifact-origin-fence"
    dyn = authority["evidence"]["dyn"]
    assert dyn["attempted"] is True
    assert dyn["passed"] is False
    assert dyn["exit_code"] == 65
    assert dyn["expected_origin"] == [-512, -512, -512]
    assert dyn["admitted_artifacts"]["grid_origin"] == [-512, 0, -512]
    assert dyn["origin_fence_rejected"] is True
    assert dyn["staging_created"] is False
    assert dyn["output_directory_exists"] is False
    assert dyn["report_published"] is False
    assert dyn["snapshots_published"] is False
    assert authority["evidence"]["source_authorization"]["consumed"] is True
    assert authority["failure"]["publication_contract"]["gate_run"] is False
    assert authority["failure"]["publication_contract"]["training_run"] is False
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False


def test_71450_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71450.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "3405dc2e648450b85beb56bc04c60dddd03b12a20d42e70a76cadbc5897d505b"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-stock-objective-admission-pre-source"
    )
    assert authority["cohort_id"] == "b2g26_final_71450"
    assert authority["declaration"]["sha256"] == (
        "d02c7c0737cf38be314394dd30e0293dcdf0b80c004efc1e0d072abc72f437c4"
    )
    assert authority["failure"]["phase"] == (
        "stock-objective-l1-admission-before-source"
    )
    stock = authority["evidence"]["stock_objective_admission"]
    assert stock["stock_root"] == (
        "/home/raymond/q2-multires-isolated/B2/stock-9f09b5e"
    )
    assert sorted(stock["completed_stock_summaries"]) == [
        f"q2dm{number}" for number in range(1, 8)
    ]
    q2dm8 = stock["q2dm8"]
    assert q2dm8["passed"] is False
    assert q2dm8["deterministic"] is True
    assert q2dm8["bsp_sha256"] == (
        "c4baf022c69334bb20c42bc113f163c16c44e443df59ccb0d66e26bc0e3f6d9b"
    )
    assert q2dm8["error"] == (
        "AtlasAnalysisError: objective entity 45 is 228.090 units from admitted L1"
    )
    assert q2dm8["objectives_published"] is False
    assert q2dm8["atlas_manifest_published"] is False
    assert q2dm8["analysis_manifest_published"] is False
    assert q2dm8["build_summary_published"] is False
    contract = authority["failure"]["publication_contract"]
    assert contract["source_stage_published"] is False
    assert contract["compiled_stage_published"] is False
    assert contract["materialized_stage_published"] is False
    assert contract["claims_stage_published"] is False
    assert contract["analysis_stage_published"] is False
    assert contract["dyn_run"] is False
    assert contract["gate_run"] is False
    assert contract["training_run"] is False
    assert authority["evidence"]["source_authorization"]["consumed"] is False
    assert authority["evidence"]["source_authorization"]["marker_exists"] is False
    assert authority["evidence"]["final_workspace"]["exists"] is False
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False
    assert authority["admission"]["source_stage_published"] is False
    assert authority["admission"]["replacement_declaration_status"] == (
        "pending-analyzer-fix-and-fresh-qualification"
    )


def test_71449_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71449.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "64eb7995394e0a1456bc054241e551bd815602abd007d9f6fb9c7e52e961c0e5"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-dyn-operator-argv-parse"
    )
    assert authority["cohort_id"] == "b2g26_final_71449"
    assert authority["failure"]["phase"] == "dyn-operator-argv-parse"
    dyn = authority["evidence"]["dyn"]
    assert dyn["attempted"] is True
    assert dyn["passed"] is False
    assert dyn["exit_code"] == 64
    assert dyn["invoked_flag"] == "--expected-origin=-512,-512,-512"
    assert dyn["parse_arguments_rejected"] is True
    assert dyn["staging_created"] is False
    assert dyn["report_published"] is False
    assert dyn["snapshots_published"] is False
    assert dyn["executable_sha256"] == (
        "7552f9948bf51c8fec9228cf5db0f42d407cced76b9b4a88fa83c7359b4da5f8"
    )
    assert authority["evidence"]["source_authorization"]["consumed"] is True
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False
    assert authority["operator_transcript"]["exit_code"] == 64
    assert authority["operator_transcript"]["stderr_first_line"] == (
        "q2-dyn-evidence: unknown flag --expected-origin=-512,-512,-512"
    )


def test_71448_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71448.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "5af6539207d41bfffe4d98404a6cc96de7b14fbc17907d3ab3f7256cf2574350"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-atlas-build-b1-client-release-closure"
    )
    assert authority["cohort_id"] == "b2g26_final_71448"
    assert authority["failure"]["phase"] == (
        "atlas-build-missing-canonical-client-release-closure"
    )
    atlas = authority["evidence"]["final_campaign"]["atlas_build"]
    assert atlas["maps_attempted"] == 28
    assert atlas["pass_count"] == 0
    assert atlas["analysis_stage_published"] is False
    assert authority["evidence"]["source_authorization"]["consumed"] is True
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False


def test_71447_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71447.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "f411e66859d3176d4ed6e0ffe24aeb809db24c1e30bf7b85ae4be9d8fbc7ce9e"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-atomic-test-publisher-clean-tree-postcondition"
    )
    assert authority["cohort_id"] == "b2g26_final_71447"
    assert authority["failure"]["phase"] == (
        "atomic-test-publisher-clean-tree-postcondition"
    )
    publisher = authority["evidence"]["test_publisher"]
    assert publisher["cargo_target_created_in_repository"] == (
        "tools/q2-dyn-evidence/target/"
    )
    assert publisher["report_published"] is False
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False


def test_71446_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71446.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "4b26c670ed54585787505cf7dfbb35bdc1830fdfbd42585a16d0484622ea306f"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-compiled-cm-operator-preflight"
    )
    assert authority["cohort_id"] == "b2g26_final_71446"
    assert authority["failure"]["phase"] == (
        "compiled-cm-operator-timeout-preflight"
    )
    cm = authority["evidence"]["compiled_cm_preflight"]
    assert cm["requested_oracle_batch_timeout_seconds"] == 3600
    assert cm["accepted_timeout_range"] == {
        "minimum_exclusive_seconds": 0,
        "maximum_inclusive_seconds": 60,
    }
    assert cm["map_oracle_invocation_count"] == 0
    assert cm["output_report_exists"] is False
    assert authority["admission"]["source_stage_published"] is True
    assert authority["admission"]["compiled_stage_published"] is True
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False


def test_71445_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71445.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "d134ddd35bb6e93f1fffa71d2b6176d402ba70c2d4242b2f55b6be40efd651af"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == "permanently-failed-first-source-freeze"
    assert authority["cohort_id"] == "b2g26_final_71445"
    assert authority["failure"]["phase"] == (
        "primary-generation-lava-reward-placement"
    )
    assert authority["evidence"]["failed_member"] == {
        "files": {},
        "map": "b2g26_pits_71445300",
        "missing_suffixes": [
            ".map", ".json", ".meta.json", ".lattice.json", ".routes.json"
        ],
        "ordinal": 12,
        "seed": 71445300,
        "style": "pits",
    }
    assert authority["evidence"]["primary_population"]["actual_file_count"] == 60
    assert authority["evidence"]["cold_population"]["actual_file_count"] == 0
    assert authority["admission"]["source_stage_published"] is False
    assert authority["admission"]["retry_under_same_declaration_allowed"] is False


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


def test_71444_terminal_failure_authority_is_canonical_and_exact() -> None:
    payload = FAILURE_71444.read_bytes()
    authority = json.loads(payload)

    assert payload == cohort.canonical_bytes(authority)
    assert hashlib.sha256(payload).hexdigest() == (
        "b709b038772e349583de4eea549ec16d6180ac820ea9ff1a4e382a0ec14ccf01"
    )
    assert authority["schema"] == registry.FAILURE_SCHEMA
    assert authority["status"] == (
        "permanently-failed-materialization-authority-preflight"
    )
    assert authority["cohort_id"] == "b2g26_final_71444"
    assert authority["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71444-DECLARATION.json",
        "sha256": "da27e96b3fe8c3719a7ff1593e37b4ac768f53a36f38c877566af495a6b551bf",
    }
    admission = authority["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["source_stage_published"] is True
    assert admission["compiled_stage_published"] is True
    for stage in ("materialized", "claims", "analysis"):
        assert admission[f"{stage}_stage_published"] is False
    evidence = authority["evidence"]
    assert evidence["source_freeze"]["report_sha256"] == (
        "0986e0c70e04c7d1a70427c0218e079b885f2bbe269b3280a81a4245c2c7c098"
    )
    assert evidence["wsl_compile"]["report_sha256"] == (
        "2a93eb8782c488768eb1c81bade03872eced3e64ad65de16eec948d614986e33"
    )
    assert evidence["compiled_cm_preflight"]["report_sha256"] == (
        "a465649db8a9dc34da0e6513ef93710416bb849049608808cdaa256e9adaf4ff"
    )
    materialization = evidence["materialization_failure"]
    assert materialization["report_sha256"] == (
        "75c4d8fd2d38d9cc7ad4fdf32b612d4d761ff9ea3b46fdf66d3ec0a367cc1962"
    )
    assert materialization["attempted_count"] == 0
    assert materialization["log_file_count"] == 0
    assert materialization["materialized_staging_path_exists"] is False
    assert materialization["materialized_publish_path_exists"] is False
    assert authority["failure"]["phase"] == (
        "materialization-authority-preflight"
    )
    publication = authority["failure"]["publication_contract"]
    assert publication["materialization_run"] is True
    assert publication["maps_materialization_attempted"] == 0
    assert publication["materialized_stage_published"] is False
    assert publication["dyn_run"] is False
    assert publication["test_campaign_run"] is False
    assert publication["deployment_run"] is False
    assert publication["training_run"] is False


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
        (
            NAMED_71444,
            "b2g26_final_71444",
            "da27e96b3fe8c3719a7ff1593e37b4ac768f53a36f38c877566af495a6b551bf",
        ),
        (
            NAMED_71445,
            "b2g26_final_71445",
            "ffa5b9ccfee0340f1bad533a23fedd103a08d14d125149d1516a2326fb8a091b",
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
        "--client-root", str(root / "client"),
        "--lithium-root", str(root / "lithium"),
        "--hook-attestation", str(root / "hook-attestation.json"),
        "--fall-oracle", str(root / "q2-fall-oracle"),
        "--packer", str(root / "q2-atlas-pack"),
        "--verifier", str(root / "q2-atlas-verify"),
    ]) == 1

    captured = capsys.readouterr()
    assert "generated Atlas campaign failed:" in captured.err
    assert "permanently retired" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()
