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
        "99c13db93a8dacb9fe24f181126b8c30203f4005fdd5e96fb0b9a165ba2168f9",
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
    assert "future-only" not in text
    assert '--basedir "$FUTURE_ROOT/assets"' not in text


def test_contract_replaces_the_hand_run_single_map_materializer() -> None:
    text = DOCS[0].read_text(encoding="utf-8")
    assert "python tools/materialize_hook_claims.py \\" not in text
    assert "Do not hand-run `q2tool`" in text
