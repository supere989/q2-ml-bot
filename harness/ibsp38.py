"""Bounded, metadata-only Quake II IBSP-38 parser.

This module deliberately does not make collision, reachability, support,
visibility, or hookability decisions.  Those remain the responsibility of the
pinned Yamagi collision and movement oracles.  The parser is limited to the
structural and catalog data needed to admit a BSP into corpus quarantine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Mapping, Sequence


IBSP_MAGIC = b"IBSP"
IBSP_VERSION = 38

LUMP_NAMES = (
    "entities",
    "planes",
    "vertices",
    "visibility",
    "nodes",
    "texinfo",
    "faces",
    "lighting",
    "leafs",
    "leaffaces",
    "leafbrushes",
    "edges",
    "surfedges",
    "models",
    "brushes",
    "brushsides",
    "pop",
    "areas",
    "areaportals",
)

HEADER_SIZE = 8 + len(LUMP_NAMES) * 8

_RECORD_SIZES = {
    "planes": 20,
    "vertices": 12,
    "nodes": 28,
    "texinfo": 76,
    "faces": 20,
    "leafs": 28,
    "leaffaces": 2,
    "leafbrushes": 2,
    "edges": 4,
    "surfedges": 4,
    "models": 48,
    "brushes": 12,
    "brushsides": 4,
    "areas": 8,
    "areaportals": 8,
}


class BspValidationError(ValueError):
    """Raised when a BSP is unsafe, unsupported, or structurally invalid."""


@dataclass(frozen=True)
class BspLimits:
    max_file_bytes: int = 128 * 1024 * 1024
    max_entity_bytes: int = 1 * 1024 * 1024
    max_entities: int = 8192
    max_entity_pairs: int = 131072
    max_token_bytes: int = 16384
    max_visibility_clusters: int = 65536


@dataclass(frozen=True)
class LumpMetadata:
    name: str
    offset: int
    length: int
    records: int | None


@dataclass(frozen=True)
class EntityMetadata:
    index: int
    classname: str
    properties: tuple[tuple[str, str], ...]

    def value(self, key: str, default: str = "") -> str:
        for candidate, value in reversed(self.properties):
            if candidate == key:
                return value
        return default


@dataclass(frozen=True)
class EntityCatalog:
    class_counts: Mapping[str, int]
    deathmatch_spawn_count: int
    spawn_classes: Mapping[str, int]
    item_classes: Mapping[str, int]
    mover_classes: Mapping[str, int]
    trigger_classes: Mapping[str, int]
    teleporter_classes: Mapping[str, int]
    brush_submodels: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict:
        return {
            "class_counts": dict(sorted(self.class_counts.items())),
            "deathmatch_spawn_count": self.deathmatch_spawn_count,
            "spawn_classes": dict(sorted(self.spawn_classes.items())),
            "item_classes": dict(sorted(self.item_classes.items())),
            "mover_classes": dict(sorted(self.mover_classes.items())),
            "trigger_classes": dict(sorted(self.trigger_classes.items())),
            "teleporter_classes": dict(sorted(self.teleporter_classes.items())),
            "brush_submodels": list(self.brush_submodels),
        }


@dataclass(frozen=True)
class ModelMetadata:
    index: int
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]
    origin: tuple[float, float, float]
    headnode: int
    first_face: int
    face_count: int


@dataclass(frozen=True)
class FaceSummary:
    count: int
    lightmapped_count: int
    unlit_count: int
    unique_lightmap_offsets: int
    texture_count: int
    texture_names: tuple[str, ...]


@dataclass(frozen=True)
class LightmapMetadata:
    byte_count: int
    sha256: str
    minimum_referenced_offset: int | None
    maximum_referenced_offset: int | None


@dataclass(frozen=True)
class VisibilityMetadata:
    cluster_count: int
    row_bytes: int
    pvs_rows_present: int
    phs_rows_present: int
    pvs_visible_min: int
    pvs_visible_max: int
    pvs_visible_mean: float
    phs_visible_min: int
    phs_visible_max: int
    phs_visible_mean: float
    sha256: str


@dataclass(frozen=True)
class BspMetadata:
    sha256: str
    byte_count: int
    version: int
    lumps: tuple[LumpMetadata, ...]
    entities: tuple[EntityMetadata, ...]
    entity_catalog: EntityCatalog
    models: tuple[ModelMetadata, ...]
    faces: FaceSummary
    lightmaps: LightmapMetadata
    visibility: VisibilityMetadata

    @property
    def model0(self) -> ModelMetadata:
        return self.models[0]

    def to_dict(self, *, include_entities: bool = False) -> dict:
        value = {
            "schema": "q2-ibsp38-metadata-v1",
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "version": self.version,
            "lumps": [asdict(lump) for lump in self.lumps],
            "entity_catalog": self.entity_catalog.to_dict(),
            "models": [asdict(model) for model in self.models],
            "faces": asdict(self.faces),
            "lightmaps": asdict(self.lightmaps),
            "visibility": asdict(self.visibility),
        }
        if include_entities:
            value["entities"] = [asdict(entity) for entity in self.entities]
        return value


def _read_source(source: bytes | bytearray | memoryview | Path | str,
                 limits: BspLimits) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    else:
        path = Path(source)
        size = path.stat().st_size
        if size > limits.max_file_bytes:
            raise BspValidationError(
                f"BSP is {size} bytes; limit is {limits.max_file_bytes}"
            )
        data = path.read_bytes()
    if len(data) > limits.max_file_bytes:
        raise BspValidationError(
            f"BSP is {len(data)} bytes; limit is {limits.max_file_bytes}"
        )
    if len(data) < HEADER_SIZE:
        raise BspValidationError("truncated IBSP-38 header")
    return data


def _parse_lumps(data: bytes) -> tuple[LumpMetadata, ...]:
    magic, version = struct.unpack_from("<4si", data, 0)
    if magic != IBSP_MAGIC:
        raise BspValidationError(f"unsupported BSP magic {magic!r}")
    if version != IBSP_VERSION:
        raise BspValidationError(
            f"unsupported IBSP version {version}; expected {IBSP_VERSION}"
        )

    lumps = []
    occupied: list[tuple[int, int, str]] = []
    for index, name in enumerate(LUMP_NAMES):
        offset, length = struct.unpack_from("<ii", data, 8 + index * 8)
        if offset < 0 or length < 0:
            raise BspValidationError(f"{name} lump has a negative range")
        end = offset + length
        if end > len(data):
            raise BspValidationError(
                f"{name} lump range [{offset}, {end}) exceeds file size {len(data)}"
            )
        if length and offset < HEADER_SIZE:
            raise BspValidationError(f"{name} lump overlaps the BSP header")
        record_size = _RECORD_SIZES.get(name)
        records = None
        if record_size is not None:
            if length % record_size:
                raise BspValidationError(
                    f"{name} lump length {length} is not a multiple of {record_size}"
                )
            records = length // record_size
        if length:
            occupied.append((offset, end, name))
        lumps.append(LumpMetadata(name, offset, length, records))

    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if current[0] < previous[1]:
            raise BspValidationError(
                f"overlapping lumps: {previous[2]} and {current[2]}"
            )
    return tuple(lumps)


def _lump_bytes(data: bytes, lumps: Sequence[LumpMetadata], name: str) -> bytes:
    lump = lumps[LUMP_NAMES.index(name)]
    return data[lump.offset:lump.offset + lump.length]


class _EntityTokenizer:
    def __init__(self, data: bytes, limits: BspLimits):
        try:
            self.text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BspValidationError("entity string is not valid UTF-8") from error
        self.index = 0
        self.limits = limits

    def _skip(self) -> None:
        while self.index < len(self.text):
            if self.text[self.index].isspace():
                self.index += 1
                continue
            if self.text.startswith("//", self.index):
                newline = self.text.find("\n", self.index + 2)
                self.index = len(self.text) if newline < 0 else newline + 1
                continue
            break

    def token(self) -> str | None:
        self._skip()
        if self.index >= len(self.text):
            return None
        character = self.text[self.index]
        if character in "{}":
            self.index += 1
            return character
        if character == '"':
            self.index += 1
            out: list[str] = []
            while self.index < len(self.text):
                character = self.text[self.index]
                self.index += 1
                if character == '"':
                    value = "".join(out)
                    if len(value.encode("utf-8")) > self.limits.max_token_bytes:
                        raise BspValidationError("entity token exceeds byte limit")
                    return value
                if character == "\\" and self.index < len(self.text):
                    escaped = self.text[self.index]
                    self.index += 1
                    out.append({"n": "\n", "t": "\t"}.get(escaped, escaped))
                else:
                    out.append(character)
            raise BspValidationError("unterminated quoted entity token")

        start = self.index
        while self.index < len(self.text):
            if self.text[self.index].isspace() or self.text[self.index] in "{}":
                break
            self.index += 1
        value = self.text[start:self.index]
        if not value:
            raise BspValidationError("invalid empty entity token")
        if len(value.encode("utf-8")) > self.limits.max_token_bytes:
            raise BspValidationError("entity token exceeds byte limit")
        return value


def _parse_entities(raw: bytes, limits: BspLimits) -> tuple[EntityMetadata, ...]:
    if len(raw) > limits.max_entity_bytes:
        raise BspValidationError(
            f"entity lump is {len(raw)} bytes; limit is {limits.max_entity_bytes}"
        )
    if not raw or raw[-1] != 0:
        raise BspValidationError("entity lump must end with one NUL terminator")
    raw = raw[:-1]
    if b"\x00" in raw:
        raise BspValidationError("entity lump contains an embedded NUL")

    tokenizer = _EntityTokenizer(raw, limits)
    entities: list[EntityMetadata] = []
    pair_count = 0
    while True:
        token = tokenizer.token()
        if token is None:
            break
        if token != "{":
            raise BspValidationError(f"expected entity '{{', got {token!r}")
        properties: list[tuple[str, str]] = []
        while True:
            key = tokenizer.token()
            if key is None:
                raise BspValidationError("unterminated entity")
            if key == "}":
                break
            if key == "{":
                raise BspValidationError("nested entity brace")
            value = tokenizer.token()
            if value is None or value in {"{", "}"}:
                raise BspValidationError(f"entity key {key!r} has no value")
            properties.append((key, value))
            pair_count += 1
            if pair_count > limits.max_entity_pairs:
                raise BspValidationError("entity key/value pair limit exceeded")
        index = len(entities)
        classname = ""
        for key, value in reversed(properties):
            if key == "classname":
                classname = value
                break
        if not classname:
            raise BspValidationError(f"entity {index} has no classname")
        entities.append(EntityMetadata(index, classname, tuple(properties)))
        if len(entities) > limits.max_entities:
            raise BspValidationError("entity count limit exceeded")

    if not entities or entities[0].classname != "worldspawn":
        raise BspValidationError("entity 0 must be worldspawn")
    return tuple(entities)


def _parse_models(raw: bytes, face_count: int, node_count: int) -> tuple[ModelMetadata, ...]:
    models = []
    for index in range(len(raw) // 48):
        values = struct.unpack_from("<9f3i", raw, index * 48)
        coordinates = values[:9]
        if not all(math.isfinite(value) for value in coordinates):
            raise BspValidationError(f"model {index} contains NaN or infinity")
        mins = tuple(coordinates[0:3])
        maxs = tuple(coordinates[3:6])
        origin = tuple(coordinates[6:9])
        if any(low > high for low, high in zip(mins, maxs)):
            raise BspValidationError(f"model {index} has inverted bounds")
        headnode, first_face, model_faces = values[9:12]
        if headnode >= node_count or headnode < -1:
            raise BspValidationError(f"model {index} has invalid headnode {headnode}")
        if first_face < 0 or model_faces < 0 or first_face + model_faces > face_count:
            raise BspValidationError(f"model {index} has invalid face range")
        models.append(ModelMetadata(
            index=index,
            mins=mins,
            maxs=maxs,
            origin=origin,
            headnode=headnode,
            first_face=first_face,
            face_count=model_faces,
        ))
    if not models:
        raise BspValidationError("model 0 is missing")
    return tuple(models)


def _decode_c_string(raw: bytes, description: str) -> str:
    value = raw.split(b"\x00", 1)[0]
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as error:
        raise BspValidationError(f"{description} is not ASCII") from error


def _parse_faces(
    raw: bytes,
    *,
    plane_count: int,
    surfedges: Sequence[int],
    edge_count: int,
    texinfo_raw: bytes,
    lighting: bytes,
) -> tuple[FaceSummary, tuple[int, ...]]:
    textures: list[str] = []
    texinfo_count = len(texinfo_raw) // 76
    for index in range(texinfo_count):
        texture = _decode_c_string(
            texinfo_raw[index * 76 + 40:index * 76 + 72],
            f"texinfo {index} texture",
        )
        next_texinfo = struct.unpack_from("<i", texinfo_raw, index * 76 + 72)[0]
        if next_texinfo < -1 or next_texinfo >= texinfo_count:
            raise BspValidationError(
                f"texinfo {index} has invalid animation link {next_texinfo}"
            )
        textures.append(texture)

    light_offsets: list[int] = []
    count = len(raw) // 20
    for index in range(count):
        (plane, _side, first_edge, num_edges, texinfo,
         style0, style1, style2, style3, light_offset) = struct.unpack_from(
            "<Hhihh4Bi", raw, index * 20
        )
        if plane >= plane_count:
            raise BspValidationError(f"face {index} has invalid plane {plane}")
        if texinfo < 0 or texinfo >= texinfo_count:
            raise BspValidationError(f"face {index} has invalid texinfo {texinfo}")
        if first_edge < 0 or num_edges < 0 or first_edge + num_edges > len(surfedges):
            raise BspValidationError(f"face {index} has invalid surfedge range")
        for surfedge in surfedges[first_edge:first_edge + num_edges]:
            if abs(surfedge) >= edge_count:
                raise BspValidationError(
                    f"face {index} references invalid edge {surfedge}"
                )
        if light_offset < -1 or light_offset >= len(lighting):
            raise BspValidationError(
                f"face {index} has invalid lightmap offset {light_offset}"
            )
        if light_offset >= 0:
            if style0 == 255:
                raise BspValidationError(
                    f"face {index} has lightdata but no active light style"
                )
            light_offsets.append(light_offset)
        elif any(style != 255 for style in (style0, style1, style2, style3)):
            # Quake compilers may retain style zero on intentionally unlit
            # faces.  This is diagnostic rather than a structural failure.
            pass

    unique_textures = tuple(sorted(set(textures)))
    return FaceSummary(
        count=count,
        lightmapped_count=len(light_offsets),
        unlit_count=count - len(light_offsets),
        unique_lightmap_offsets=len(set(light_offsets)),
        texture_count=len(unique_textures),
        texture_names=unique_textures,
    ), tuple(light_offsets)


def _validate_nodes_leafs(
    *,
    nodes_raw: bytes,
    leafs_raw: bytes,
    leaffaces_raw: bytes,
    leafbrushes_raw: bytes,
    plane_count: int,
    face_count: int,
    brush_count: int,
    visibility_clusters: int,
) -> None:
    node_count = len(nodes_raw) // 28
    leaf_count = len(leafs_raw) // 28
    for index in range(node_count):
        values = struct.unpack_from("<3i6h2H", nodes_raw, index * 28)
        plane = values[0]
        children = values[1:3]
        first_face, num_faces = values[-2:]
        if plane < 0 or plane >= plane_count:
            raise BspValidationError(f"node {index} has invalid plane {plane}")
        for child in children:
            if child >= 0:
                if child >= node_count:
                    raise BspValidationError(f"node {index} has invalid child {child}")
            elif -1 - child >= leaf_count:
                raise BspValidationError(f"node {index} has invalid leaf child {child}")
        if first_face + num_faces > face_count:
            raise BspValidationError(f"node {index} has invalid face range")

    leaffaces = struct.unpack(f"<{len(leaffaces_raw) // 2}H", leaffaces_raw) \
        if leaffaces_raw else ()
    leafbrushes = struct.unpack(f"<{len(leafbrushes_raw) // 2}H", leafbrushes_raw) \
        if leafbrushes_raw else ()
    for index in range(leaf_count):
        values = struct.unpack_from("<ihh6h4H", leafs_raw, index * 28)
        cluster = values[1]
        first_face, num_faces, first_brush, num_brushes = values[-4:]
        if cluster < -1 or cluster >= visibility_clusters:
            raise BspValidationError(f"leaf {index} has invalid cluster {cluster}")
        if first_face + num_faces > len(leaffaces):
            raise BspValidationError(f"leaf {index} has invalid leafface range")
        if first_brush + num_brushes > len(leafbrushes):
            raise BspValidationError(f"leaf {index} has invalid leafbrush range")
    if any(face >= face_count for face in leaffaces):
        raise BspValidationError("leaffaces contains an invalid face index")
    if any(brush >= brush_count for brush in leafbrushes):
        raise BspValidationError("leafbrushes contains an invalid brush index")


def _decode_vis_row(raw: bytes, offset: int, row_bytes: int, label: str) -> int:
    cursor = offset
    emitted = 0
    visible = 0
    while emitted < row_bytes:
        if cursor >= len(raw):
            raise BspValidationError(f"truncated {label} visibility row")
        value = raw[cursor]
        cursor += 1
        if value:
            emitted += 1
            if emitted > row_bytes:
                raise BspValidationError(f"overlong {label} visibility row")
            visible += value.bit_count()
            continue
        if cursor >= len(raw):
            raise BspValidationError(f"truncated {label} zero run")
        run = raw[cursor]
        cursor += 1
        if run == 0:
            raise BspValidationError(f"zero-length {label} visibility run")
        emitted += run
        if emitted > row_bytes:
            raise BspValidationError(f"overlong {label} visibility zero run")
    return visible


def _visibility_metadata(raw: bytes, limits: BspLimits) -> VisibilityMetadata:
    digest = hashlib.sha256(raw).hexdigest()
    if not raw:
        return VisibilityMetadata(0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0.0, digest)
    if len(raw) < 4:
        raise BspValidationError("truncated visibility header")
    cluster_count = struct.unpack_from("<i", raw, 0)[0]
    if cluster_count < 0 or cluster_count > limits.max_visibility_clusters:
        raise BspValidationError(f"invalid visibility cluster count {cluster_count}")
    table_end = 4 + cluster_count * 8
    if table_end > len(raw):
        raise BspValidationError("truncated visibility offset table")
    row_bytes = (cluster_count + 7) // 8
    pvs: list[int] = []
    phs: list[int] = []
    for cluster in range(cluster_count):
        pvs_offset, phs_offset = struct.unpack_from("<ii", raw, 4 + cluster * 8)
        for offset, destination, name in (
            (pvs_offset, pvs, "PVS"),
            (phs_offset, phs, "PHS"),
        ):
            if offset == -1:
                continue
            if offset < table_end or offset >= len(raw):
                raise BspValidationError(
                    f"cluster {cluster} has invalid {name} offset {offset}"
                )
            destination.append(_decode_vis_row(raw, offset, row_bytes, name))

    def stats(values: Sequence[int]) -> tuple[int, int, float]:
        if not values:
            return 0, 0, 0.0
        return min(values), max(values), sum(values) / len(values)

    pvs_min, pvs_max, pvs_mean = stats(pvs)
    phs_min, phs_max, phs_mean = stats(phs)
    return VisibilityMetadata(
        cluster_count=cluster_count,
        row_bytes=row_bytes,
        pvs_rows_present=len(pvs),
        phs_rows_present=len(phs),
        pvs_visible_min=pvs_min,
        pvs_visible_max=pvs_max,
        pvs_visible_mean=pvs_mean,
        phs_visible_min=phs_min,
        phs_visible_max=phs_max,
        phs_visible_mean=phs_mean,
        sha256=digest,
    )


def _catalog_entities(
    entities: Sequence[EntityMetadata], model_count: int
) -> EntityCatalog:
    class_counts = Counter(entity.classname for entity in entities)
    spawn_counts = Counter()
    item_counts = Counter()
    mover_counts = Counter()
    trigger_counts = Counter()
    teleporter_counts = Counter()
    submodels: list[Mapping[str, object]] = []

    for entity in entities:
        classname = entity.classname
        if classname.startswith("info_player_"):
            spawn_counts[classname] += 1
        if classname.startswith(("weapon_", "ammo_", "item_", "key_")):
            item_counts[classname] += 1
        if classname.startswith("func_"):
            mover_counts[classname] += 1
        if classname.startswith("trigger_"):
            trigger_counts[classname] += 1
        if "teleport" in classname:
            teleporter_counts[classname] += 1

        model = entity.value("model")
        if model.startswith("*"):
            try:
                model_index = int(model[1:])
            except ValueError as error:
                raise BspValidationError(
                    f"entity {entity.index} has invalid brush model {model!r}"
                ) from error
            if model_index <= 0 or model_index >= model_count:
                raise BspValidationError(
                    f"entity {entity.index} references missing submodel {model_index}"
                )
            submodels.append({
                "entity_index": entity.index,
                "model_index": model_index,
                "classname": classname,
                "target": entity.value("target"),
                "targetname": entity.value("targetname"),
            })

    return EntityCatalog(
        class_counts=dict(sorted(class_counts.items())),
        deathmatch_spawn_count=spawn_counts["info_player_deathmatch"],
        spawn_classes=dict(sorted(spawn_counts.items())),
        item_classes=dict(sorted(item_counts.items())),
        mover_classes=dict(sorted(mover_counts.items())),
        trigger_classes=dict(sorted(trigger_counts.items())),
        teleporter_classes=dict(sorted(teleporter_counts.items())),
        brush_submodels=tuple(submodels),
    )


def parse_ibsp38(
    source: bytes | bytearray | memoryview | Path | str,
    *,
    limits: BspLimits = BspLimits(),
) -> BspMetadata:
    """Parse and validate non-collision metadata from one classic Q2 BSP."""

    data = _read_source(source, limits)
    lumps = _parse_lumps(data)

    entities = _parse_entities(_lump_bytes(data, lumps, "entities"), limits)
    planes_raw = _lump_bytes(data, lumps, "planes")
    nodes_raw = _lump_bytes(data, lumps, "nodes")
    leafs_raw = _lump_bytes(data, lumps, "leafs")
    leaffaces_raw = _lump_bytes(data, lumps, "leaffaces")
    leafbrushes_raw = _lump_bytes(data, lumps, "leafbrushes")
    edges_raw = _lump_bytes(data, lumps, "edges")
    surfedges_raw = _lump_bytes(data, lumps, "surfedges")
    models_raw = _lump_bytes(data, lumps, "models")
    brushes_raw = _lump_bytes(data, lumps, "brushes")
    faces_raw = _lump_bytes(data, lumps, "faces")
    texinfo_raw = _lump_bytes(data, lumps, "texinfo")
    lighting = _lump_bytes(data, lumps, "lighting")

    plane_count = len(planes_raw) // 20
    node_count = len(nodes_raw) // 28
    face_count = len(faces_raw) // 20
    edge_count = len(edges_raw) // 4
    brush_count = len(brushes_raw) // 12
    surfedges = struct.unpack(f"<{len(surfedges_raw) // 4}i", surfedges_raw) \
        if surfedges_raw else ()

    visibility = _visibility_metadata(
        _lump_bytes(data, lumps, "visibility"), limits
    )
    models = _parse_models(models_raw, face_count, node_count)
    faces, light_offsets = _parse_faces(
        faces_raw,
        plane_count=plane_count,
        surfedges=surfedges,
        edge_count=edge_count,
        texinfo_raw=texinfo_raw,
        lighting=lighting,
    )
    _validate_nodes_leafs(
        nodes_raw=nodes_raw,
        leafs_raw=leafs_raw,
        leaffaces_raw=leaffaces_raw,
        leafbrushes_raw=leafbrushes_raw,
        plane_count=plane_count,
        face_count=face_count,
        brush_count=brush_count,
        visibility_clusters=visibility.cluster_count,
    )
    catalog = _catalog_entities(entities, len(models))
    lightmaps = LightmapMetadata(
        byte_count=len(lighting),
        sha256=hashlib.sha256(lighting).hexdigest(),
        minimum_referenced_offset=min(light_offsets) if light_offsets else None,
        maximum_referenced_offset=max(light_offsets) if light_offsets else None,
    )

    return BspMetadata(
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        version=IBSP_VERSION,
        lumps=lumps,
        entities=entities,
        entity_catalog=catalog,
        models=models,
        faces=faces,
        lightmaps=lightmaps,
        visibility=visibility,
    )


def stock_inventory_record(map_name: str, metadata: BspMetadata) -> dict:
    """Return the bounded, coordinate-light stock-fixture inventory record."""

    if not map_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in map_name
    ):
        raise ValueError(f"invalid canonical map name {map_name!r}")
    catalog = metadata.entity_catalog
    return {
        "canonical_id": map_name,
        "bsp_sha256": metadata.sha256,
        "bsp_bytes": metadata.byte_count,
        "ibsp_version": metadata.version,
        "entity_count": len(metadata.entities),
        "deathmatch_spawn_count": catalog.deathmatch_spawn_count,
        "item_classes": dict(sorted(catalog.item_classes.items())),
        "model_count": len(metadata.models),
        "submodel_entity_count": len(catalog.brush_submodels),
        "face_count": metadata.faces.count,
        "lightmapped_face_count": metadata.faces.lightmapped_count,
        "lightdata_bytes": metadata.lightmaps.byte_count,
        "visibility_clusters": metadata.visibility.cluster_count,
    }
