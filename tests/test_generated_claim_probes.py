from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.generated_claim_probes import (
    GeneratedClaimProbeError,
    _project_route_point,
    _trigger_hurt_runtime_bounds,
    analyze_non_hook_claims,
    generated_bsp_provenance,
    load_generator_claims,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _claims_fixture(tmp_path: Path) -> tuple[Path, dict]:
    name = "generated_fixture"
    sources = {
        "map_sha256": (f"{name}.map", b"map source\n"),
        "meta_sha256": (f"{name}.meta.json", b"{}\n"),
        "lattice_sha256": (f"{name}.lattice.json", b"{}\n"),
        "hook_zones_sha256": (f"{name}.json", b"# hooks\n"),
        "hook_materialization_sha256": (
            f"{name}.hook-materialization.json", b"{}\n"
        ),
        "routes_sha256": (f"{name}.routes.json", b"{}\n"),
    }
    source_hashes = {}
    for key, (filename, data) in sources.items():
        (tmp_path / filename).write_bytes(data)
        source_hashes[key] = _digest(data)
    claims = {
        "schema": "q2-generator-claims-v3",
        "map": name,
        "generator": "v6",
        "source_files": source_hashes,
        "spawns": [
            {"claim_id": f"spawn:{index:04d}", "origin_milliunits": [index * 400_000, 0, 24_000]}
            for index in range(8)
        ],
        "hazard_claims": [
            {
                "claim_id": "hazard:hurt:0000", "type": "hurt",
                "bounds_milliunits": [-64_000, -64_000, -128_000, 64_000, 64_000, -64_000],
            },
            {
                "claim_id": "hazard:lava:0000", "type": "lava",
                "bounds_milliunits": [64_000, 0, 0, 128_000, 64_000, 64_000],
            },
        ],
        "hook_claims": [
            {
                "claim_id": f"hook:{index:04d}:candidate:0000",
                "source_milliunits": [index * 32_000, 0, 24_000],
                "trace_target_milliunits": [index * 32_000 + 16_000, 0, 200_000],
                "measured_anchor_milliunits": [index * 32_000 + 16_000, 0, 200_000],
                "landing_milliunits": [index * 32_000 + 16_000, 0, 24_000],
                "release_after_ticks": 2,
                "distance_milliunits": round((16_000 ** 2 + 154_000 ** 2) ** 0.5),
                "flags": 1,
            }
            for index in range(6)
        ],
        "route_claims": [
            {
                "claim_id": "route:0000:segment:0000", "route_id": "route:0000",
                "source_milliunits": [8_000, 8_000, 24_000],
                "target_milliunits": [24_000, 8_000, 24_000],
            },
            {
                "claim_id": "route:0000:segment:0001", "route_id": "route:0000",
                "source_milliunits": [24_000, 8_000, 24_000],
                "target_milliunits": [8_000, 8_000, 24_000],
            },
        ],
        "routes": [{
            "route_id": "route:0000", "archetype": "balanced", "claimed_cost_q8": 8192,
            "segment_claim_ids": [
                "route:0000:segment:0000", "route:0000:segment:0001",
            ],
        }],
    }
    path = tmp_path / f"{name}.generator-claims.json"
    path.write_bytes(_canonical(claims))
    return path, claims


class _Entity:
    classname = "trigger_hurt"
    index = 1

    def __init__(self, *, origin: str = "", spawnflags: str = "") -> None:
        self.origin = origin
        self.spawnflags = spawnflags

    def value(self, key: str) -> str:
        return {
            "model": "*1", "origin": self.origin, "spawnflags": self.spawnflags,
        }.get(key, "")


class _WorldEntity:
    classname = "worldspawn"

    def value(self, key: str) -> str:
        return {
            "_ml_min_floor_light_value": "650",
            "_ml_min_interior_light_value": "800",
            "_ml_min_interior_light_radius": "320",
        }.get(key, "")


class _LightEntity:
    classname = "light"
    index = 2

    def __init__(self, *, origin: str = "32 32 128", intensity: str = "900", radius: str = "448"):
        self.origin = origin
        self.intensity = intensity
        self.radius = radius

    def value(self, key: str) -> str:
        return {
            "origin": self.origin, "light": self.intensity,
            "_ml_radius": self.radius, "_ml_floor_light": "1",
            "_ml_region": "floor_0_0_0",
        }.get(key, "")


class _FakeCm:
    def __init__(
        self, *, lava: bool = True, projection: bool = True,
        blocked_inward_samples: set[int] | None = None,
    ) -> None:
        self.lava = lava
        self.projection = projection
        self.blocked_inward_samples = blocked_inward_samples or set()
        self.requests: list[dict] = []

    def call(self, requests: list[dict]) -> list[dict]:
        self.requests.extend(deepcopy(requests))
        output = []
        for request in requests:
            identifier = request["id"]
            if identifier.startswith("claim-hurt"):
                output.append({"startsolid": True, "allsolid": True})
            elif identifier.startswith("claim-lava-trace"):
                output.append({
                    "fraction": 0.5 if self.lava else 1.0,
                    "contents": 8 if self.lava else 0,
                    "endpos": [96.0, 32.0, 20.0],
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                })
            elif identifier.startswith("claim-lava-contents"):
                output.append({"contents": 8 if self.lava else 0})
            elif identifier.startswith("route-support"):
                output.append({
                    "fraction": 0.5 if self.projection else 1.0,
                    "startsolid": False,
                    "endpos": [request["start"][0], request["start"][1], 24.0],
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                })
            elif identifier.startswith("lighting"):
                output.append({"fraction": 1.0, "startsolid": False})
            elif identifier.startswith("lethal-inward"):
                sample_index = int(identifier.rsplit(":", 1)[1])
                output.append({
                    "fraction": 0.5,
                    "startsolid": sample_index in self.blocked_inward_samples,
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                })
            elif identifier.startswith("lethal-void"):
                output.append({"fraction": 1.0, "startsolid": False})
            elif identifier.startswith("lethal-guard"):
                output.append({"startsolid": True, "allsolid": True})
            else:
                raise AssertionError(identifier)
        return output


def _compiled_fixture() -> tuple[SimpleNamespace, dict, list, list]:
    metadata = SimpleNamespace(
        entities=[_WorldEntity(), _Entity(), _LightEntity()],
        models=[
            SimpleNamespace(mins=(-256, -256, -256), maxs=(256, 256, 256), headnode=0),
            SimpleNamespace(mins=(-64, -64, -128), maxs=(64, 64, -64), headnode=7),
        ],
        lightmaps=SimpleNamespace(byte_count=4096, sha256="a" * 64),
        faces=SimpleNamespace(lightmapped_count=32),
    )
    nodes = {
        (0, 0, 1): SimpleNamespace(position=(8.0, 8.0, 24.0)),
        (1, 0, 1): SimpleNamespace(position=(24.0, 8.0, 24.0)),
    }
    edges = [
        {"source": [0, 0, 1], "target": [1, 0, 1], "cost": 4096, "evidence": 1, "validation_version": 1},
        {"source": [1, 0, 1], "target": [0, 0, 1], "cost": 4096, "evidence": 1, "validation_version": 1},
    ]
    spawns = [
        {
            "entity_ordinal": 3, "region_id": 1,
            "origin_milliunits": [8_000, 8_000, 24_000],
        },
        {
            "entity_ordinal": 4, "region_id": 1,
            "origin_milliunits": [24_000, 8_000, 24_000],
        },
    ]
    return metadata, nodes, edges, spawns


def test_route_projection_starts_at_pmove_quantum_under_low_overhang() -> None:
    class LowOverhangCm:
        def __init__(self) -> None:
            self.requests = []

        def call(self, requests):
            self.requests.extend(requests)
            if requests[0]["id"].startswith("route-support:"):
                assert len(requests) == 1
                return [{
                    "fraction": 0.001462,
                    "startsolid": False,
                    "allsolid": False,
                    "endpos": [2368.0, 1856.0, 216.03125],
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                }]
            assert all(
                request["id"].startswith("route-connector:")
                for request in requests
            )
            return [
                {
                    "fraction": 1,
                    "startsolid": False,
                    "allsolid": False,
                    "endpos": request["end"],
                    "plane": {"normal": [0.0, 0.0, 0.0]},
                }
                for request in requests
            ]

    cm = LowOverhangCm()
    aliased_key = (180, 116, 45)
    expected_key = (179, 115, 45)
    # The first-writer representative is 17 units away in the nominal cell.
    # Three adjacent representatives are tied at sqrt(128) and exact
    # bidirectional connector evidence selects stable (z, y, x) order.
    nodes = {
        aliased_key: SimpleNamespace(position=(2376.0, 1871.0, 216.03125)),
        expected_key: SimpleNamespace(position=(2360.0, 1848.0, 216.03125)),
        (180, 115, 45): SimpleNamespace(position=(2376.0, 1848.0, 216.03125)),
        (179, 116, 45): SimpleNamespace(position=(2360.0, 1864.0, 216.03125)),
    }

    projected = _project_route_point(
        cm, [2_368_000, 1_856_000, 216_000], nodes,
        (-512, 0, -512), "route:0001:segment:0005:target",
    )

    assert projected == expected_key
    assert cm.requests[0]["start"] == [2368.0, 1856.0, 216.125]
    assert cm.requests[0]["end"] == [2368.0, 1856.0, 152.0]
    assert cm.requests[1]["start"] == [2368.0, 1856.0, 216.03125]
    assert cm.requests[1]["end"] == [2360.0, 1848.0, 216.03125]
    assert cm.requests[2]["start"] == [2360.0, 1848.0, 216.03125]
    assert cm.requests[2]["end"] == [2368.0, 1856.0, 216.03125]


def test_route_projection_rejects_one_way_neighbor_connector() -> None:
    class OneWayCm:
        def call(self, requests):
            if requests[0]["id"].startswith("route-support:"):
                return [{
                    "fraction": 0.001,
                    "startsolid": False, "allsolid": False,
                    "endpos": [704.0, 1728.0, 24.03125],
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                }]
            return [
                {
                    "fraction": 1 if index % 2 == 0 else 0.5,
                    "startsolid": False, "allsolid": False,
                }
                for index, _ in enumerate(requests)
            ]

    nodes = {
        (75, 139, 33): SimpleNamespace(
            position=(696.0, 1720.0, 24.03125),
        ),
    }

    with pytest.raises(
        GeneratedClaimProbeError,
        match="no compiled connector to a nearby L1 origin",
    ):
        _project_route_point(
            OneWayCm(), [704_000, 1_728_000, 24_000], nodes,
            (-512, -512, -512), "route:0002:segment:0032:target",
        )


def _safety_fixture() -> dict:
    return {
        "version": 1, "guard_height": 96, "guard_thickness": 16,
        "lethal_edges": [{"side": "west", "segment": [0, -32, 0, 32, 64]}],
        "guard_walls": [[0, -32, 64, 16, 32, 160]],
    }


def _l0_semantics_fixture() -> dict[str, int]:
    return {"hurt:1:raw": 144, "hurt:1:expanded": 384}


def test_strict_load_provenance_and_non_hook_challenges(tmp_path: Path) -> None:
    path, expected = _claims_fixture(tmp_path)
    claims, digest = load_generator_claims(path, "generated_fixture")
    assert claims == expected
    bsp = tmp_path / "generated_fixture.bsp"
    bsp.write_bytes(b"IBSP compiled fixture")
    provenance = generated_bsp_provenance(bsp, claims, digest)
    assert provenance["generator_claims_sha256"] == digest
    assert provenance["bsp_sha256"] == _digest(bsp.read_bytes())

    metadata, nodes, edges, spawns = _compiled_fixture()
    cm = _FakeCm()
    result = analyze_non_hook_claims(
        cm, metadata, nodes, edges, (0, 0, 0), spawns, claims,
        _safety_fixture(), _l0_semantics_fixture(),
    )
    assert [item["status"] for item in result["hazard_claims"]] == ["oracle", "oracle"]
    assert [item["cost_q8"] for item in result["route_claims"]] == [4096, 4096]
    assert result["lighting"]["lightdata_sha256"] == "a" * 64
    assert result["lighting"]["floor_light_region_count"] == 1
    assert result["lighting"]["floor_light_region_ids"] == ["floor_0_0_0"]
    assert result["lighting"]["spawn_nav_region_count"] == 1
    assert result["lighting"]["dark_spawns"] == []
    assert "hooks" not in result
    requests = {request["id"]: request for request in cm.requests}
    assert requests["lethal-inward:0:0"]["start"] == [32.125, 0.0, 112.0]
    assert requests["lethal-inward:0:0"]["end"] == [32.125, 0.0, 0.0]
    assert requests["lethal-guard:0:0"]["start"] == [8.0, -25.6, 72.0]


def test_generator_claim_loader_rejects_retired_v2_schema(tmp_path: Path) -> None:
    path, claims = _claims_fixture(tmp_path)
    claims["schema"] = "q2-generator-claims-v2"
    path.write_bytes(_canonical(claims))

    with pytest.raises(GeneratedClaimProbeError, match="not frozen v6"):
        load_generator_claims(path, "generated_fixture")


def test_lethal_floor_uses_a_bounded_alternate_segment_witness(tmp_path: Path) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    cm = _FakeCm(blocked_inward_samples={0})
    result = analyze_non_hook_claims(
        cm, metadata, nodes, edges, (0, 0, 0), spawns, claims,
        _safety_fixture(), _l0_semantics_fixture(),
    )
    assert result["hazards"]["lethal_drop_edges"] == 1
    requests = {request["id"]: request for request in cm.requests}
    assert requests["lethal-void:0"]["start"][:2] == [-32.0, -25.6]


@pytest.mark.parametrize(
    ("side", "segment", "wall", "inward", "guard"),
    [
        ("west", [0, -32, 0, 32, 64], [0, -32, 64, 16, 32, 160],
         [32.125, 0.0, 112.0], [8.0, -25.6, 72.0]),
        ("east", [0, -32, 0, 32, 64], [-16, -32, 64, 0, 32, 160],
         [-32.125, 0.0, 112.0], [-8.0, -25.6, 72.0]),
        ("south", [-32, 0, 32, 0, 64], [-32, 0, 64, 32, 16, 160],
         [0.0, 32.125, 112.0], [-25.6, 8.0, 72.0]),
        ("north", [-32, 0, 32, 0, 64], [-32, -16, 64, 32, 0, 160],
         [0.0, -32.125, 112.0], [-25.6, -8.0, 72.0]),
    ],
)
def test_lethal_probe_preserves_world_units_and_side_orientation(
    tmp_path: Path, side: str, segment: list[int], wall: list[int],
    inward: list[float], guard: list[float],
) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    safety = {
        "version": 1, "guard_height": 96, "guard_thickness": 16,
        "lethal_edges": [{"side": side, "segment": segment}],
        "guard_walls": [wall],
    }
    cm = _FakeCm()
    analyze_non_hook_claims(
        cm, metadata, nodes, edges, (0, 0, 0), spawns, claims, safety,
        _l0_semantics_fixture(),
    )
    requests = {request["id"]: request for request in cm.requests}
    assert requests["lethal-inward:0:0"]["start"] == inward
    assert requests["lethal-guard:0:0"]["start"] == guard


def test_lethal_floor_rejects_when_every_segment_witness_is_blocked(
    tmp_path: Path,
) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    with pytest.raises(GeneratedClaimProbeError, match="no compiled interior floor"):
        analyze_non_hook_claims(
            _FakeCm(blocked_inward_samples=set(range(11))),
            metadata, nodes, edges, (0, 0, 0), spawns, claims,
            _safety_fixture(), _l0_semantics_fixture(),
        )


def test_trigger_hurt_uses_exact_runtime_linked_aabb_law() -> None:
    model = SimpleNamespace(mins=(100, 0, 0), maxs=(164, 64, 16))
    raw, linked, standing = _trigger_hurt_runtime_bounds(model, (0, 0, 0))
    assert raw == [100_000, 0, 0, 164_000, 64_000, 16_000]
    assert linked == [98_000, -2_000, -2_000, 166_000, 66_000, 18_000]
    assert standing == [81_000, -19_000, -35_000, 183_000, 83_000, 43_000]


def test_trigger_hurt_runtime_bounds_bind_entity_origin() -> None:
    model = SimpleNamespace(mins=(100, 0, 0), maxs=(164, 64, 16))
    raw, linked, standing = _trigger_hurt_runtime_bounds(model, (16, -8, 4))
    assert raw == [116_000, -8_000, 4_000, 180_000, 56_000, 20_000]
    assert linked == [114_000, -10_000, 2_000, 182_000, 58_000, 22_000]
    assert standing == [97_000, -27_000, -31_000, 199_000, 75_000, 47_000]


def test_hurt_claim_rejects_missing_retained_sparse_atlas_evidence(
    tmp_path: Path,
) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    with pytest.raises(
        GeneratedClaimProbeError,
        match="no retained sparse Atlas hurt evidence",
    ):
        analyze_non_hook_claims(
            _FakeCm(), metadata, nodes, edges, (0, 0, 0), spawns, claims,
            _safety_fixture(), {},
        )


@pytest.mark.parametrize("spawnflags", ["1", "2", "3"])
def test_stateful_trigger_hurt_is_potential_unknown(
    tmp_path: Path, spawnflags: str,
) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    metadata.entities[1] = _Entity(spawnflags=spawnflags)
    result = analyze_non_hook_claims(
        _FakeCm(), metadata, nodes, edges, (0, 0, 0), spawns, claims,
        _safety_fixture(), _l0_semantics_fixture(),
    )
    hurt = result["hazard_claims"][0]
    assert hurt["status"] == "unknown"
    assert hurt["contained"] is False
    lava = result["hazard_claims"][1]
    assert lava["status"] == "oracle"
    assert lava["contained"] is True


def test_claims_and_probes_fail_closed(tmp_path: Path) -> None:
    path, claims = _claims_fixture(tmp_path)
    path.write_bytes(json.dumps(claims, indent=2).encode())
    with pytest.raises(GeneratedClaimProbeError, match="not canonical"):
        load_generator_claims(path)

    path.write_bytes(_canonical(claims))
    (tmp_path / "generated_fixture.map").write_bytes(b"changed\n")
    with pytest.raises(GeneratedClaimProbeError, match="digest differs"):
        load_generator_claims(path)

    metadata, nodes, edges, spawns = _compiled_fixture()
    with pytest.raises(GeneratedClaimProbeError, match="no compiled lava hit"):
        analyze_non_hook_claims(
            _FakeCm(lava=False), metadata, nodes, edges, (0, 0, 0), spawns, claims,
            _safety_fixture(), _l0_semantics_fixture(),
        )
    with pytest.raises(GeneratedClaimProbeError, match="no evidenced Atlas route"):
        analyze_non_hook_claims(
            _FakeCm(), metadata, nodes, edges[:1], (0, 0, 0), spawns, claims,
            _safety_fixture(), _l0_semantics_fixture(),
        )

    unlit = deepcopy(metadata)
    unlit.lightmaps = SimpleNamespace(byte_count=0, sha256="a" * 64)
    with pytest.raises(GeneratedClaimProbeError, match="no admitted qrad lighting"):
        analyze_non_hook_claims(
            _FakeCm(), unlit, nodes, edges, (0, 0, 0), spawns, claims,
            _safety_fixture(), _l0_semantics_fixture(),
        )


def test_low_intensity_light_is_rejected(tmp_path: Path) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    metadata.entities[-1] = _LightEntity(intensity="649")
    with pytest.raises(GeneratedClaimProbeError, match="no qualified v6 light"):
        analyze_non_hook_claims(
            _FakeCm(), metadata, nodes, edges, (0, 0, 0), spawns, claims,
            _safety_fixture(), _l0_semantics_fixture(),
        )


def test_far_light_cannot_clear_spawn_regions(tmp_path: Path) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    metadata.entities[-1] = _LightEntity(origin="5000 5000 128", radius="448")
    result = analyze_non_hook_claims(
        _FakeCm(), metadata, nodes, edges, (0, 0, 0), spawns, claims,
        _safety_fixture(), _l0_semantics_fixture(),
    )
    assert result["lighting"]["dark_spawns"] == [
        {
            "entity_ordinal": 3,
            "nav_region_id": 1,
            "eligible_light_entity_ordinals": [],
        },
        {
            "entity_ordinal": 4,
            "nav_region_id": 1,
            "eligible_light_entity_ordinals": [],
        },
    ]


def test_removed_lethal_guard_is_rejected(tmp_path: Path) -> None:
    _, claims = _claims_fixture(tmp_path)
    metadata, nodes, edges, spawns = _compiled_fixture()
    safety = _safety_fixture()
    safety["guard_walls"] = []
    with pytest.raises(GeneratedClaimProbeError, match="no exact guard proposal"):
        analyze_non_hook_claims(
            _FakeCm(), metadata, nodes, edges, (0, 0, 0), spawns, claims, safety,
            _l0_semantics_fixture(),
        )
