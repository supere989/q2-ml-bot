from __future__ import annotations

from pathlib import Path

from harness.atlas_source_closure import (
    ATLAS_ANALYZER_FIXED_INPUTS,
    atlas_analyzer_authority_inputs,
    atlas_analyzer_authority_sha256,
)
from tools.generator_claim_validator import _expected_analyzer_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_analyzer_and_promotion_share_one_complete_source_closure() -> None:
    inputs = atlas_analyzer_authority_inputs(ROOT)
    relative = {path.relative_to(ROOT).as_posix() for path in inputs}
    rust = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "crates/q2-lattice/src").rglob("*.rs")
    }

    assert set(ATLAS_ANALYZER_FIXED_INPUTS).issubset(relative)
    assert rust
    assert rust.issubset(relative)
    assert "harness/atlas_source_closure.py" in relative
    assert _expected_analyzer_sha256() == atlas_analyzer_authority_sha256(ROOT)
