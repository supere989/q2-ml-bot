from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "docs/multires/B2-C-GENERATOR-CLAIM-CONTRACT.md",
    ROOT / "docs/multires/B2-GATE-ASSEMBLY.md",
)
B1_ROOT = "/home/raymond/q2-multires-isolated/B1-authorities-909b1e46"
B1_MANIFEST_SHA256 = (
    "8d163d87a6919fc5d7f3761b17aa1aeaae7e71a5c505b80392a315802e11a92f"
)
B1_FILENAMES = (
    "B1-GATE.json",
    "CONTENT-MANIFEST.json",
    "hook-parity-pullspeed-1700.json",
    "q2-cm-oracle",
    "q2-fall-oracle",
    "q2-hook-oracle",
    "q2-pmove-oracle",
)


@pytest.mark.parametrize("path", DOCS)
def test_current_producer_contract_is_complete_and_fail_closed(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    for required in (
        "python tools/compile_generated_cohort.py",
        "--source-root",
        "--staging-root",
        "--publish-root",
        "--log-root",
        "--q2tool",
        '--basedir "$FUTURE_ROOT/assets/baseq2"',
        "pak0.pak",
        "pics/colormap.pcx",
        "tools/materialize_generated_cohort.py",
        "--compiled-dir",
        "--stage-dir",
        "--materialized-dir",
        "--log-dir",
        "--cm-oracle",
        "--pmove-oracle",
        "--hook-oracle",
        "--fall-oracle",
        "--hook-parity-attestation",
        "-bsp -vis -fast -rad -bounce 0 -threads 1 -basedir",
        "declaration order",
        "168 files",
        "196 files",
        "renameat2(RENAME_NOREPLACE)",
        "terminal",
        "non-reusable",
        "absent",
        "no retry",
        B1_ROOT,
        B1_MANIFEST_SHA256,
        "b2g26_final_71439",
        "B2-GENERATED-COHORT-71439-FAILURE.json",
        "fbcbca7c134c2d2595ab98cfe939f615b226cab4a5e28e836f824d41e4f76255",
        "fc6435e81ac1d10f8a32602169df68cc34103c4b64a2cdbcf96be55260a3733d",
        "b171b2ee4ab02f8b960684544e49471dcfc5e11cdef105687a77938e1dcafe69",
        "harness/atlas_analyzer.py:5404",
        "tools/check_python_syntax_floor.py",
        "Python 3.10",
        "/home/raymond/miniconda3/bin/python3.11",
        "b25abf001748dc7ebb4b25013b2572d4e6913246b4c3b8e8b726b3da45494ff4",
        "zstandard 0.19.0",
        "import pytest, zstandard, torch",
        "python -m pytest --version",
        "only after generated promotion and Dyn",
        "permanently retired",
        "B2-GENERATED-COHORT-71440-FAILURE.json",
        "2abbb7c9de511fd4b497111317d61be439f37c96702441d6d7190e9afb5cf19c",
        "94681d77f53b0514a2795865d593b6007d58bef9e9bbf1be0a7ef2f16d7e46b1",
        "11689967027196a77443d02628da1ee72df33bfa71475a1967634e268f47afc4",
        "B2-GENERATED-COHORT-71441-FAILURE.json",
        "c241b81b458eb525334a720e9059902dabef30347195ba1200d63b530133f3e3",
        "292e0e483c66596bfba58972bdf0e58ed36d938b3412c8868a3b2c10ba510aa3",
        "B2-GENERATED-COHORT-71442-FAILURE.json",
        "Cohort 71442 is permanently retired",
        "b2g26_final_71443",
        "d890e151cbc3446622a8c0f5fdd2bd23352583c6431e1484262587c3c7246713",
        "B2-GENERATED-COHORT-71443-FAILURE.json",
        "Cohort 71443 is permanently retired",
        "da89be636079b0cc38583281113002f0578d2608c5a31af052fca8c03d05f723",
        "6e748dd45bfd013cfd9c57f2ec60289b9abf40da946e511d771efe096d02a456",
        "c0c7f8c857e8ef60f0f74b959fef6b34f458fc69223146d7245ce2e79de76d84",
        "196d25d0de40e4333dda9fe4c946e84ae571133554cb72e5ffa1c835bef1bb2d",
        "six collection errors",
        "zero executed tests",
        "publisher-ordering",
        "99c13db93a8dacb9fe24f181126b8c30203f4005fdd5e96fb0b9a165ba2168f9",
        "b2q26_275d4fa_71623700",
        "09bd298d87739515d468f432219eefcad01e8586a87a71339f5121900a6f57c5",
        "eb99e08e5934d281556b0b6584ab23fe236adb8fce81f1cc7045229b368b9a25",
        "275d4fa646ccf2c64ba8628cd4aa8b21644fa90d",
        "7bd808b2194a44b80dc64fb88c700209d4657e9a",
        "b2g26_final_71444",
        "B2-GENERATED-COHORT-71444-DECLARATION.json",
        "da27e96b3fe8c3719a7ff1593e37b4ac768f53a36f38c877566af495a6b551bf",
        "71444000..71444003",
        "71444600..71444603",
        "Exactly one immutable/no-retry final producer attempt was authorized",
        "The first source-generation invocation consumes the sole authorization",
        "B2-GENERATED-COHORT-71444-FAILURE.json",
        "Cohort 71444 is permanently retired",
        "b2q26_3b17223_71625100",
        "351baccaabf405e0ef240c1def18e4ede796ff417e73230524e9f0f9b0c0491b",
        "58295d227ddd3694a0ddae5af46e2bbc98cc60dbe6b6751b4e42df01c06b1cd6",
        "3b17223ab32e20152aead1eb32a79e239d6f4d8a",
        "fa2b106d19dbb115e6acd4c344b3820b3013464a",
        "b2g26_final_71445",
        "B2-GENERATED-COHORT-71445-DECLARATION.json",
        "ffa5b9ccfee0340f1bad533a23fedd103a08d14d125149d1516a2326fb8a091b",
        "71445000..71445003",
        "71445600..71445603",
        "B2-GENERATED-COHORT-71445-FAILURE.json",
        "Cohort 71445 is permanently retired",
        "d134ddd35bb6e93f1fffa71d2b6176d402ba70c2d4242b2f55b6be40efd651af",
        "cf87d90e7f7d40a9baae7e5bf54c27491f26d4a28531830f4a5cc79e4add1db7",
        "2167bfdef17cf247e329e5761dc7e44d3c22d34f5a3181faea5b8c2f737ee8a3",
        "could not place a unique lava-rim reward",
        "b2q26_a05ddb7_71626100",
        "69e2b1979feae22c706839dc24f8923b60e34d5b623c8f03b0e5ebb51181a549",
        "a05ddb7037774c1b246a6b13972b228570acb8ef",
        "01c27fc60da4ae6f2aedd6138c50dabfcd866525",
        "fb71a121d05dc02ad4d634f537abb331ed7d4ea29da0e5c3199afe8c0b442001",
        "b2g26_final_71446",
        "B2-GENERATED-COHORT-71446-DECLARATION.json",
        "58d52bd958249a70bf8115ab1c442fb6888a6d69b290a636303986f69acb658f",
        "71446000..71446003",
        "71446600..71446603",
        "B2-GENERATED-COHORT-71446-FAILURE.json",
        "Cohort 71446 is permanently retired",
        "b2g26_final_71447",
        "B2-GENERATED-COHORT-71447-DECLARATION.json",
        "76c0ffc41ff80cb4b9f0ea6648240a73b55f0a7933970f8f2e2fd05a086cb4aa",
        "71447000..71447003",
        "71447600..71447603",
        "b2q26_74628f1_71804000",
        "48e7f3488addacbd43d6c5f6b6fe92f35a62b3c3f5d717a3c646816858bd7e73",
        "B2-GENERATED-COHORT-71447-FAILURE.json",
        "Cohort 71447 is permanently retired",
        "f411e66859d3176d4ed6e0ffe24aeb809db24c1e30bf7b85ae4be9d8fbc7ce9e",
        "tools/q2-dyn-evidence/target/",
        "q2-b2-test-report-v2",
        "CARGO_TARGET_DIR",
        "build.target-dir",
        "4b26c670ed54585787505cf7dfbb35bdc1830fdfbd42585a16d0484622ea306f",
        "oracle batch timeout must be finite and in (0, 60]",
        "not a 3,600-second runtime timeout",
        "b709b038772e349583de4eea549ec16d6180ac820ea9ff1a4e382a0ec14ccf01",
        "0986e0c70e04c7d1a70427c0218e079b885f2bbe269b3280a81a4245c2c7c098",
        "2a93eb8782c488768eb1c81bade03872eced3e64ad65de16eec948d614986e33",
        "a465649db8a9dc34da0e6513ef93710416bb849049608808cdaa256e9adaf4ff",
        "75c4d8fd2d38d9cc7ad4fdf32b612d4d761ff9ea3b46fdf66d3ec0a367cc1962",
        "materialization authority preflight",
        "tools/preflight_b2_materialization_authorities.py",
        "Qualification artifacts",
        "final producer lane itself is strictly sequential",
        "Tests never overlap compilation",
        "qualification",
        "non-admissible",
        "fe4b86bbb0ab331dca4f7fd1418106c69ba4d4ea34b36774cb7e9259d27502bc",
        "5929532e0edae77b48073abccf4a4f3afdbacfb6905d1eadfb7f18d1dc5ba151",
        "fresh B1",
        "not a cohort artifact",
    ):
        assert required in text, f"{path.name} is missing {required!r}"
    for filename in B1_FILENAMES:
        assert filename in text

    assert "Cohort 71440" in text
    assert "permanently retired" in text
    assert "b2g26_final_71441" in text
    assert "b2g26_final_71442" in text
    assert "b2g26_final_71443" in text
    assert "Cohort 71443 is permanently retired" in text
    assert "b2g26_final_71444" in text
    assert "Cohort 71444 is permanently retired" in text
    assert "b2g26_final_71445" in text
    assert "Retired 71445 final attempt" in text
    assert "Cohort 71445 is permanently retired" in text
    assert "b2g26_final_71446" in text
    assert "Retired 71446 final attempt" in text
    assert "Cohort 71446 is permanently retired" in text
    assert "b2g26_final_71447" in text
    assert "Retired 71447 final attempt" in text
    assert "Cohort 71447 is permanently retired" in text
    assert "Exactly one immutable/no-retry final producer attempt was authorized" in text
    assert "future-only" not in text
    assert '--basedir "$FUTURE_ROOT/assets"' not in text


def test_current_gate_contract_retires_71448_and_clears_active_authority() -> None:
    text = (ROOT / "docs/multires/B2-GATE-ASSEMBLY.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "b2g26_final_71447",
        "B2-GENERATED-COHORT-71447-DECLARATION.json",
        "76c0ffc41ff80cb4b9f0ea6648240a73b55f0a7933970f8f2e2fd05a086cb4aa",
        "b2q26_74628f1_71804000",
        "48e7f3488addacbd43d6c5f6b6fe92f35a62b3c3f5d717a3c646816858bd7e73",
        "B2-GENERATED-COHORT-71447-FAILURE.json",
        "f411e66859d3176d4ed6e0ffe24aeb809db24c1e30bf7b85ae4be9d8fbc7ce9e",
        "q2-b2-test-report-v2",
        "b2g26_final_71448",
        "B2-GENERATED-COHORT-71448-DECLARATION.json",
        "0b48462a8cd8dfb752a73b711954616dd22d45d857748d316505bd17c976262a",
        "71448000..71448003",
        "71448600..71448603",
        "b2q26_ae41232_71805000",
        "c7a623eed20eea7c115c6167391158be90bb70bd4914e1d591ecee9c1f2ff3d8",
        "Retired 71448 final attempt",
        "B2-GENERATED-COHORT-71448-FAILURE.json",
        "5af6539207d41bfffe4d98404a6cc96de7b14fbc17907d3ab3f7256cf2574350",
        "atlas-build-missing-canonical-client-release-closure",
        "permanently-failed-atlas-build-b1-client-release-closure",
        "source-only staged q2-ml-client root",
        "release/q2-cm-oracle",
        "release/q2-pmove-oracle",
        "Cohort 71448 is permanently retired",
        "`ACTIVE_FINAL_AUTHORITY = None`",
        "there is currently no active final cohort",
        "historical and non-executable until a successor activation",
        "Lithium hook oracle",
        "byte identity of the CM, Pmove, and hook oracles",
        "separately committed immutable declaration",
        "fresh green disposable qualification",
        "The assembler rejects declarations for retired cohorts 71426 through 71448",
    ):
        assert required in text
    for stale_claim in (
        "Only the exact fresh 71446 declaration is eligible",
        "The active 71446 prerequisite",
        "current alias names fresh cohort 71446 and can execute",
        "The active final `COHORT_ID` is `b2g26_final_71447`",
        "The active final `COHORT_ID` is `b2g26_final_71448`",
        "Active 71448 final cohort",
        "`ACTIVE_FINAL_AUTHORITY` explicitly pins",
        "sole active eligible declaration pair",
        "This activation authorizes",
        "The active authority and schema both pin 71448",
    ):
        assert stale_claim not in text


def test_contract_replaces_the_hand_run_single_map_materializer() -> None:
    text = DOCS[0].read_text(encoding="utf-8")
    assert "python tools/materialize_hook_claims.py \\" not in text
    assert "Do not hand-run `q2tool`" in text
