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
        "python tools/materialize_generated_cohort.py",
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
        "authority-bound",
        "not a cohort artifact",
    ):
        assert required in text, f"{path.name} is missing {required!r}"
    for filename in B1_FILENAMES:
        assert filename in text

    assert "No replacement cohort is authorized" not in text
    assert "future-only" not in text
    assert '--basedir "$FUTURE_ROOT/assets"' not in text


def test_contract_replaces_the_hand_run_single_map_materializer() -> None:
    text = DOCS[0].read_text(encoding="utf-8")
    assert "python tools/materialize_hook_claims.py \\" not in text
    assert "Do not hand-run `q2tool`" in text
