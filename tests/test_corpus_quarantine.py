from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import zipfile

import pytest

from harness.corpus_quarantine import (
    ArchiveLimits,
    ProvenanceRecord,
    QuarantineError,
    admit_corpus,
    classify_duplicates,
    duplicate_signature,
    inventory_stock_pak,
    load_provenance,
    quarantine_archive,
)
from harness.ibsp38 import (
    BspLimits,
    BspValidationError,
    HEADER_SIZE,
    LUMP_NAMES,
    parse_ibsp38,
)


def _entities(*, message="fixture", second_model=False):
    model = '\n{\n"classname" "func_door"\n"model" "*1"\n}\n' \
        if second_model else ""
    return (
        '{\n"classname" "worldspawn"\n"message" "' + message + '"\n}\n'
        '{\n"classname" "info_player_deathmatch"\n"origin" "0 0 24"\n}\n'
        '{\n"classname" "weapon_rocketlauncher"\n"origin" "64 0 24"\n}\n'
        + model
    ).encode() + b"\x00"


def build_bsp(*, message="fixture", include_model0=True, second_model=False):
    lumps = {name: b"" for name in LUMP_NAMES}
    lumps["entities"] = _entities(message=message, second_model=second_model)
    lumps["planes"] = struct.pack("<4fi", 0.0, 0.0, 1.0, 0.0, 2)
    lumps["vertices"] = b"".join((
        struct.pack("<3f", 0.0, 0.0, 0.0),
        struct.pack("<3f", 64.0, 0.0, 0.0),
        struct.pack("<3f", 0.0, 64.0, 0.0),
    ))
    lumps["visibility"] = struct.pack("<i2i2B", 1, 12, 13, 1, 1)
    lumps["nodes"] = struct.pack(
        "<3i6h2H", 0, -1, -1, -64, -64, -64, 64, 64, 64, 0, 1
    )
    texture = b"fixture/stone".ljust(32, b"\x00")
    lumps["texinfo"] = struct.pack(
        "<8fii32si", *(0.0,) * 8, 0, 0, texture, -1
    )
    lumps["faces"] = struct.pack(
        "<Hhihh4Bi", 0, 0, 0, 3, 0, 0, 255, 255, 255, 0
    )
    lumps["lighting"] = b"\x80\x80\x80"
    lumps["leafs"] = struct.pack(
        "<ihh6h4H", 0, 0, 0, -64, -64, -64, 64, 64, 64, 0, 1, 0, 0
    )
    lumps["leaffaces"] = struct.pack("<H", 0)
    lumps["edges"] = struct.pack("<6H", 0, 1, 1, 2, 2, 0)
    lumps["surfedges"] = struct.pack("<3i", 0, 1, 2)
    if include_model0:
        models = [struct.pack(
            "<9f3i",
            -64.0, -64.0, -24.0, 64.0, 64.0, 64.0,
            0.0, 0.0, 0.0, 0, 0, 1,
        )]
        if second_model:
            models.append(struct.pack(
                "<9f3i",
                -16.0, -16.0, -16.0, 16.0, 16.0, 16.0,
                0.0, 0.0, 0.0, 0, 0, 1,
            ))
        lumps["models"] = b"".join(models)
    lumps["areas"] = struct.pack("<2i", 0, 0)

    header = bytearray(struct.pack("<4si", b"IBSP", 38))
    cursor = HEADER_SIZE
    body = bytearray()
    ranges = []
    for name in LUMP_NAMES:
        content = lumps[name]
        ranges.append((cursor if content else 0, len(content)))
        body.extend(content)
        cursor += len(content)
    for offset, length in ranges:
        header.extend(struct.pack("<2i", offset, length))
    return bytes(header + body)


def write_zip(path: Path, members, *, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, content in members:
            archive.writestr(name, content)


def write_pak(path: Path, members):
    body = bytearray()
    directory = bytearray()
    offset = 12
    for name, content in members:
        encoded = name.encode("ascii")
        assert len(encoded) < 56
        body.extend(content)
        directory.extend(struct.pack("<56sii", encoded, offset, len(content)))
        offset += len(content)
    path.write_bytes(
        b"PACK" + struct.pack("<2i", 12 + len(body), len(directory))
        + body + directory
    )


def provenance(report, member="maps/q2dm1.bsp", canonical="q2dm1", aliases=()):
    return ProvenanceRecord(
        canonical_id=canonical,
        bsp_member=member,
        aliases=tuple(aliases),
        source_url=None,
        manual_origin="locally installed stock fixture",
        author="id Software",
        license_name="commercial game data",
        license_evidence="analysis-only local installation; no redistribution grant",
        redistribution="analysis-only",
        archive_sha256=report.archive_sha256,
        bsp_sha256=report.bsp_metadata[member].sha256,
    )


def test_ibsp38_metadata_catalogs_entities_models_faces_lightmaps_and_pvs():
    metadata = parse_ibsp38(build_bsp(second_model=True))

    assert metadata.version == 38
    assert metadata.entity_catalog.deathmatch_spawn_count == 1
    assert metadata.entity_catalog.item_classes == {"weapon_rocketlauncher": 1}
    assert metadata.entity_catalog.mover_classes == {"func_door": 1}
    assert metadata.entity_catalog.brush_submodels == ({
        "entity_index": 3,
        "model_index": 1,
        "classname": "func_door",
        "target": "",
        "targetname": "",
    },)
    assert len(metadata.models) == 2
    assert metadata.faces.count == 1
    assert metadata.faces.lightmapped_count == 1
    assert metadata.faces.texture_names == ("fixture/stone",)
    assert metadata.lightmaps.byte_count == 3
    assert metadata.visibility.cluster_count == 1
    assert metadata.visibility.pvs_visible_mean == 1.0


def test_ibsp38_rejects_malformed_lump_range():
    data = bytearray(build_bsp())
    entities_offset_field = 8
    struct.pack_into("<2i", data, entities_offset_field, len(data) - 2, 100)
    with pytest.raises(BspValidationError, match="exceeds file"):
        parse_ibsp38(data)


def test_ibsp38_rejects_missing_model0():
    with pytest.raises(BspValidationError, match="model 0 is missing"):
        parse_ibsp38(build_bsp(include_model0=False))


def test_ibsp38_rejects_oversized_entity_string():
    with pytest.raises(BspValidationError, match="entity lump"):
        parse_ibsp38(
            build_bsp(), limits=BspLimits(max_entity_bytes=16)
        )


@pytest.mark.parametrize("member", ("../evil.bsp", "/absolute.bsp", "C:/evil.bsp"))
def test_zip_quarantine_rejects_path_traversal_and_absolute_paths(tmp_path, member):
    archive = tmp_path / "bad.zip"
    write_zip(archive, [(member, build_bsp())])
    with pytest.raises(QuarantineError, match="unsafe|absolute"):
        quarantine_archive(archive)


def test_zip_quarantine_rejects_symlink(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        info = zipfile.ZipInfo("maps/link.bsp")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(info, "q2dm1.bsp")
    with pytest.raises(QuarantineError, match="symlink"):
        quarantine_archive(archive)


def test_zip_quarantine_rejects_case_collision(tmp_path):
    archive = tmp_path / "bad.zip"
    write_zip(archive, [("maps/A.bsp", build_bsp()), ("maps/a.bsp", build_bsp())])
    with pytest.raises(QuarantineError, match="case-colliding"):
        quarantine_archive(archive)


@pytest.mark.parametrize(
    "name,content,match",
    (
        ("install.sh", b"echo unsafe", "script"),
        ("readme.txt", b"\x7fELF" + b"\x00" * 20, "magic"),
        ("nested.pk3", b"PK\x03\x04", "nested"),
    ),
)
def test_zip_quarantine_rejects_active_and_nested_payloads(
    tmp_path, name, content, match
):
    archive = tmp_path / "bad.zip"
    write_zip(archive, [(name, content)])
    with pytest.raises(QuarantineError, match=match):
        quarantine_archive(archive)


def test_zip_quarantine_rejects_decompression_ratio(tmp_path):
    archive = tmp_path / "bomb.zip"
    write_zip(
        archive,
        [("docs/repeated.txt", b"A" * 100_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    with pytest.raises(QuarantineError, match="compression ratio"):
        quarantine_archive(
            archive, limits=ArchiveLimits(max_compression_ratio=2.0)
        )


def test_clean_zip_requires_hash_bound_provenance(tmp_path):
    archive = tmp_path / "clean.zip"
    write_zip(archive, [("maps/q2dm1.bsp", build_bsp())])
    report = quarantine_archive(archive)

    entries = admit_corpus(report, [provenance(report, aliases=("base1",))])

    assert entries[0].canonical_id == "q2dm1"
    assert entries[0].aliases == ("base1",)
    assert report.assets.by_class == {"bsp": 1}
    assert report.assets.unresolved_texture_references == ("fixture/stone",)
    with pytest.raises(QuarantineError, match="archive SHA-256"):
        admit_corpus(report, [replace(
            provenance(report), archive_sha256="0" * 64
        )])


def test_provenance_rejects_unsafe_alias(tmp_path):
    archive = tmp_path / "clean.zip"
    write_zip(archive, [("maps/q2dm1.bsp", build_bsp())])
    report = quarantine_archive(archive)

    with pytest.raises(QuarantineError, match="alias"):
        admit_corpus(report, [provenance(report, aliases=("../escape",))])


def test_provenance_rejects_case_folded_alias_collision(tmp_path):
    archive = tmp_path / "clean.zip"
    write_zip(archive, [
        ("maps/alpha.bsp", build_bsp(message="alpha")),
        ("maps/beta.bsp", build_bsp(message="beta")),
    ])
    report = quarantine_archive(archive)
    records = [
        provenance(report, "maps/alpha.bsp", "alpha", aliases=("shared",)),
        provenance(report, "maps/beta.bsp", "shared"),
    ]

    with pytest.raises(QuarantineError, match="duplicate canonical ID or alias"):
        admit_corpus(report, records)


def test_clean_pak_admits_selected_bsp_without_extraction(tmp_path):
    archive = tmp_path / "pak0.pak"
    write_pak(archive, [
        ("maps/q2dm1.bsp", build_bsp()),
        ("textures/fixture/stone.wal", b"texture"),
        ("docs/readme.txt", b"fixture"),
    ])

    report = quarantine_archive(archive, include_bsp=("maps/q2dm1.bsp",))
    entries = admit_corpus(report, [provenance(report)])

    assert report.archive_format == "quake-pak"
    assert len(report.members) == 3
    assert len(entries) == 1
    assert report.assets.unresolved_texture_references == ()


def test_cli_runs_directly_from_outside_repository(tmp_path):
    archive = tmp_path / "clean.zip"
    output = tmp_path / "report.json"
    write_zip(archive, [("maps/q2dm1.bsp", build_bsp())])
    script = Path(__file__).parents[1] / "tools" / "corpus_quarantine.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(archive), "--output", str(output)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["quarantine"]["archive_format"] == "zip"


def test_exact_alias_and_near_duplicate_classification(tmp_path):
    exact = build_bsp(message="same")
    near = build_bsp(message="slightly changed metadata")
    archive = tmp_path / "maps.zip"
    write_zip(archive, [
        ("maps/alpha.bsp", exact),
        ("maps/alpha-alias.bsp", exact),
        ("maps/alpha-near.bsp", near),
    ])
    report = quarantine_archive(archive)
    records = [
        provenance(report, "maps/alpha.bsp", "alpha"),
        provenance(report, "maps/alpha-alias.bsp", "alpha_alias"),
        provenance(report, "maps/alpha-near.bsp", "alpha_near"),
    ]
    classifications = classify_duplicates(admit_corpus(report, records))
    by_pair = {(item.left, item.right): item for item in classifications}

    assert by_pair[("alpha", "alpha_alias")].kind == "exact"
    assert by_pair[("alpha", "alpha_alias")].canonical_id == "alpha"
    assert by_pair[("alpha", "alpha_near")].kind == "near"
    signature = duplicate_signature(
        report.bsp_metadata["maps/alpha.bsp"]
    )
    encoded = json.dumps(
        {key: value for key, value in signature.items() if key != "signature_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert signature["signature_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert "origin" not in json.dumps(signature)


def test_stock_inventory_is_complete_and_analysis_only():
    fixture = Path(__file__).parent / "fixtures" / "corpus" / "stock-q2dm1-q2dm8.json"
    value = json.loads(fixture.read_text())

    assert value["schema"] == "q2-stock-map-fixtures-v1"
    assert [entry["canonical_id"] for entry in value["maps"]] == [
        f"q2dm{index}" for index in range(1, 9)
    ]
    assert all(entry["ibsp_version"] == 38 for entry in value["maps"])
    assert all(entry["deathmatch_spawn_count"] >= 2 for entry in value["maps"])
    assert all(entry["redistribution"] == "analysis-only" for entry in value["maps"])
    assert all(len(entry["bsp_sha256"]) == 64 for entry in value["maps"])


def test_local_stock_pak_matches_pinned_fixture_and_provenance():
    default_pak = Path.home() / ".local/share/YamagiQ2/baseq2/pak1.pak"
    pak = Path(os.environ.get("Q2_STOCK_PAK", default_pak))
    if not pak.is_file():
        pytest.skip("locally installed retail pak1.pak is unavailable")

    fixture_path = (
        Path(__file__).parent / "fixtures" / "corpus"
        / "stock-q2dm1-q2dm8.json"
    )
    provenance_path = (
        Path(__file__).parents[1] / "docs" / "multires"
        / "stock-q2dm1-q2dm8.provenance.json"
    )
    fixture = json.loads(fixture_path.read_text())
    report = inventory_stock_pak(pak)
    entries = admit_corpus(report, load_provenance(provenance_path))

    assert report.archive_sha256 == fixture["archive_sha256"]
    assert report.archive_bytes == fixture["archive_bytes"]
    assert [entry.canonical_id for entry in entries] == [
        f"q2dm{index}" for index in range(1, 9)
    ]
    for expected in fixture["maps"]:
        metadata = report.bsp_metadata[expected["archive_member"]]
        assert metadata.sha256 == expected["bsp_sha256"]
        assert len(metadata.entities) == expected["entity_count"]
        assert metadata.entity_catalog.deathmatch_spawn_count == expected[
            "deathmatch_spawn_count"
        ]
        assert len(metadata.models) == expected["model_count"]
        assert len(metadata.entity_catalog.brush_submodels) == expected[
            "submodel_entity_count"
        ]
        assert metadata.faces.count == expected["face_count"]
        assert metadata.faces.lightmapped_count == expected[
            "lightmapped_face_count"
        ]
        assert metadata.lightmaps.byte_count == expected["lightdata_bytes"]
        assert metadata.visibility.cluster_count == expected[
            "visibility_clusters"
        ]
        assert metadata.entity_catalog.item_classes == expected["item_classes"]
