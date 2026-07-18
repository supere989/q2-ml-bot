from __future__ import annotations

from pathlib import Path

import pytest

from harness.atlas_b1_authority import canonical_cm_physics_identity
from tools import b2_qualification_toolchain as authority_module
from tools.b2_qualification_toolchain import (
    ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
    ToolchainAuthorityError,
    inspect_baseq2_assets,
    inspect_q2tool,
    load_toolchain_authority,
)
from tools import run_b2_compiled_boundary_qualification as boundary


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_manifest_and_fixture_geometry_are_independently_bound() -> None:
    authority = load_toolchain_authority()
    assert authority.manifest_sha256 == ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256
    records = boundary.fixture_records(boundary.DEFAULT_FIXTURE_DIR, authority)
    assert [row["source"]["sha256"] for row in records] == [
        fixture["sha256"] for fixture in authority.fixtures
    ]
    assert [row["geometry"]["ceiling_bottom_units"] for row in records] == [
        104, 105, 106,
    ]


def test_manifest_byte_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = tmp_path / "authority.json"
    mutated.write_bytes(
        (ROOT / authority_module.MANIFEST_RELATIVE_PATH).read_bytes()
        .replace(b'"-fast"', b'"-full"')
    )
    monkeypatch.setattr(authority_module, "MANIFEST_RELATIVE_PATH", mutated)
    with pytest.raises(ToolchainAuthorityError, match="digest differs"):
        load_toolchain_authority()


def test_caller_bytes_cannot_redefine_q2tool_or_basedir(
    tmp_path: Path,
) -> None:
    authority = load_toolchain_authority()
    fake_q2tool = tmp_path / "q2tool"
    fake_q2tool.write_bytes(b"caller-selected q2tool")
    with pytest.raises(ToolchainAuthorityError, match="q2tool bytes differ"):
        inspect_q2tool(fake_q2tool, authority)
    assets_parent = tmp_path / "assets"
    assets_parent.mkdir()
    with pytest.raises(ToolchainAuthorityError, match="baseq2 itself"):
        inspect_baseq2_assets(assets_parent, authority)


def test_geometry_parser_uses_brush_planes_not_boundary_comment() -> None:
    source = (
        ROOT / "tests/fixtures/compiled_boundary/spawn_ceiling_104.map"
    ).read_text(encoding="ascii")
    # Leave the authoritative comment saying 104, but lower the actual ceiling
    # brush and wall-top planes. The parser must report the authored planes.
    mutated = source.replace(" 104 )", " 103 )").replace(" 120 )", " 119 )")
    geometry = boundary._parse_fixture_geometry(mutated, "mutated")
    assert geometry["ceiling_bottom_units"] == 103
    assert geometry["floor_top_units"] == 0


def test_cm_physics_identity_has_an_independent_frozen_vector() -> None:
    assert canonical_cm_physics_identity("00" * 32, "ff" * 32) == (
        "a51a11ba225d94f59280acd1365f06443fe4994a0b556259026628b77d7a39be"
    )
