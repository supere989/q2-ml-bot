"""Canonical toolchain authority for disposable B2 qualification.

The tracked manifest is a repository trust root, not a caller-selected pin.
Every producer loads the fixed repository path and verifies both its externally
accepted digest and its exact contents before it inspects executable or asset
bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE_PATH = Path(
    "docs/multires/B2-QUALIFICATION-TOOLCHAIN-AUTHORITY.json"
)
MANIFEST_SCHEMA = "q2-b2-qualification-toolchain-authority-v1"
ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256 = (
    "44961966343c9d1979def8afdf302202d82a98f8489ba252564e7f26a8170645"
)
Q2TOOL_SHA256 = (
    "a13dd3095ff56ca668e94c8992c915be669f9404162c9c87ded3b922316b26f0"
)
PAK0_SHA256 = (
    "1ce99eb11e7e251ccdf690858effba79836dbe5e32a4083ad00a13ecda491679"
)
COLORMAP_SHA256 = (
    "2662f321c8b229aa4ef64fb6dc8cbcb79696977cf5ccf77a96283d8b5ea9d4f7"
)
REQUIRED_PAK_MEMBER = "pics/colormap.pcx"
Q2TOOL_FLAGS = (
    "-bsp", "-vis", "-fast", "-rad", "-bounce", "0", "-threads", "1",
    "-basedir",
)
BOUNDARY_FIXTURES = (
    {
        "case_id": "spawn_ceiling_104",
        "ceiling_bottom_units": 104,
        "floor_top_units": 0,
        "relative_path": "tests/fixtures/compiled_boundary/spawn_ceiling_104.map",
        "sha256": "ecdd245f82f468196c5c0fe0510648057c2cc8f9a885902be8723780a2166995",
        "spawn_origin_units": [0, 0, 24],
    },
    {
        "case_id": "spawn_ceiling_105",
        "ceiling_bottom_units": 105,
        "floor_top_units": 0,
        "relative_path": "tests/fixtures/compiled_boundary/spawn_ceiling_105.map",
        "sha256": "2d8c9fe3e09f1b5c7ade03661a227dae81fd1eb0711e1c74147fae24ddb2f058",
        "spawn_origin_units": [0, 0, 24],
    },
    {
        "case_id": "spawn_ceiling_106",
        "ceiling_bottom_units": 106,
        "floor_top_units": 0,
        "relative_path": "tests/fixtures/compiled_boundary/spawn_ceiling_106.map",
        "sha256": "db42bdd2612d39c216633d922ed3772edcc9e5a9cc26066d3a867c61444f333b",
        "spawn_origin_units": [0, 0, 24],
    },
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ToolchainAuthorityError(ValueError):
    """The canonical manifest, executable, fixtures, or assets differ."""


@dataclass(frozen=True)
class ToolchainAuthority:
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: int
    q2tool_sha256: str
    q2tool_flags: tuple[str, ...]
    pak0_sha256: str
    colormap_sha256: str
    fixtures: tuple[Mapping[str, Any], ...]

    def manifest_record(self) -> dict[str, Any]:
        return {
            "bytes": self.manifest_bytes,
            "sha256": self.manifest_sha256,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file_record(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ToolchainAuthorityError(f"cannot stat {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ToolchainAuthorityError(f"required regular file is absent: {path}")
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolchainAuthorityError(f"duplicate manifest key {key!r}")
        result[key] = value
    return result


def _expected_manifest() -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "q2tool": {
            "sha256": Q2TOOL_SHA256,
            "flags": list(Q2TOOL_FLAGS),
        },
        "assets": {
            "pak0": {
                "sha256": PAK0_SHA256,
                "required_member": {
                    "path": REQUIRED_PAK_MEMBER,
                    "sha256": COLORMAP_SHA256,
                },
            },
        },
        "boundary_fixtures": [dict(item) for item in BOUNDARY_FIXTURES],
    }


def load_toolchain_authority(repo_root: Path = ROOT) -> ToolchainAuthority:
    repo_root = repo_root.resolve()
    if repo_root != ROOT.resolve():
        raise ToolchainAuthorityError(
            "toolchain authority must come from the repository containing its loader"
        )
    path = repo_root / MANIFEST_RELATIVE_PATH
    record = regular_file_record(path)
    raw = path.read_bytes()
    if record["sha256"] != ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256:
        raise ToolchainAuthorityError("toolchain authority manifest digest differs")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ToolchainAuthorityError(
            f"toolchain authority manifest is not JSON: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise ToolchainAuthorityError("toolchain authority manifest must be an object")
    manifest = dict(value)
    if raw != _canonical_bytes(manifest):
        raise ToolchainAuthorityError("toolchain authority manifest is not canonical")
    if manifest != _expected_manifest():
        raise ToolchainAuthorityError("toolchain authority manifest contents differ")
    for fixture in BOUNDARY_FIXTURES:
        fixture_path = repo_root / str(fixture["relative_path"])
        if regular_file_record(fixture_path)["sha256"] != fixture["sha256"]:
            raise ToolchainAuthorityError(
                f"canonical boundary fixture bytes differ: {fixture['case_id']}"
            )
    return ToolchainAuthority(
        manifest_path=path,
        manifest_sha256=record["sha256"],
        manifest_bytes=record["bytes"],
        q2tool_sha256=Q2TOOL_SHA256,
        q2tool_flags=Q2TOOL_FLAGS,
        pak0_sha256=PAK0_SHA256,
        colormap_sha256=COLORMAP_SHA256,
        fixtures=tuple(dict(item) for item in BOUNDARY_FIXTURES),
    )


def inspect_q2tool(path: Path, authority: ToolchainAuthority) -> dict[str, Any]:
    record = regular_file_record(path)
    if record["sha256"] != authority.q2tool_sha256:
        raise ToolchainAuthorityError("q2tool bytes differ from canonical authority")
    return record


def inspect_baseq2_assets(
    basedir: Path, authority: ToolchainAuthority,
) -> dict[str, Any]:
    basedir = basedir.absolute()
    if basedir.is_symlink() or not basedir.is_dir():
        raise ToolchainAuthorityError("basedir must be an existing baseq2 directory")
    if basedir.name.casefold() != "baseq2":
        raise ToolchainAuthorityError(
            "basedir must name baseq2 itself, not its parent directory"
        )
    pak = basedir / "pak0.pak"
    record = regular_file_record(pak)
    if record["sha256"] != authority.pak0_sha256:
        raise ToolchainAuthorityError("pak0.pak bytes differ from canonical authority")
    with pak.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ToolchainAuthorityError("pak0.pak header is truncated")
        magic, offset, length = struct.unpack("<4sii", header)
        if (
            magic != b"PACK" or offset < 12 or length < 0 or length % 64
            or offset + length > record["bytes"]
        ):
            raise ToolchainAuthorityError("pak0.pak directory is invalid")
        stream.seek(offset)
        directory = stream.read(length)
    matches = []
    for start in range(0, len(directory), 64):
        raw_name, member_offset, member_bytes = struct.unpack(
            "<56sii", directory[start:start + 64]
        )
        try:
            name = raw_name.split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise ToolchainAuthorityError("pak0.pak member name is not ASCII") from error
        if (
            member_offset < 12 or member_bytes < 0
            or member_offset + member_bytes > record["bytes"]
        ):
            raise ToolchainAuthorityError("pak0.pak member range is invalid")
        if name.replace("\\", "/").casefold() == REQUIRED_PAK_MEMBER:
            with pak.open("rb") as stream:
                stream.seek(member_offset)
                payload = stream.read(member_bytes)
            matches.append({
                "path": name,
                "bytes": member_bytes,
                "sha256": _sha256_bytes(payload),
            })
    if len(matches) != 1:
        raise ToolchainAuthorityError(
            "pak0.pak must contain one pics/colormap.pcx"
        )
    if matches[0]["sha256"] != authority.colormap_sha256:
        raise ToolchainAuthorityError(
            "pics/colormap.pcx bytes differ from canonical authority"
        )
    return {
        "basedir": str(basedir),
        "pak0": record,
        "required_member": matches[0],
    }
