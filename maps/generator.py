#!/usr/bin/env python3
"""
generator.py — Procedural Q2 deathmatch map generator with hook-zone annotations.

Generates .map source files compilable by qbsp/vis/light, plus a .json sidecar
that the ML bridge reads to fill the hook_zones[] observation fields.

Layout algorithm:
  - 2D room graph on a NxN grid
  - Connected open training floor inside a skybox
  - Arenas get cover blocks and platforms for evasion/hook practice
  - Eight separated starts support six-player deathmatch without forced reuse

Usage:
    python generator.py                   # one random map
    python generator.py --seed 42         # deterministic
    python generator.py --count 10        # batch
    python generator.py --name mymap      # named output
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ── Q2 world constants ────────────────────────────────────────────────────────

GRID_SIZE   = 512          # world units per grid cell
WALL_T      = 16           # brush wall thickness
PLAYER_H    = 56           # player standing height
PLAYER_XY_HALF = 16        # Quake II player bbox half-width
PLAYER_MINS_Z = -24        # origin-relative standing bbox bottom
PLAYER_MAXS_Z = 32         # origin-relative standing bbox top
PLAYER_DIAMETER = PLAYER_XY_HALF * 2
PMOVE_FIXED_QUANTUM = 0.125  # pmove_state_t origin is stored in eighth-units
# A 56u Quake II standing hull can enter a gap that is still too cramped for
# reliable movement, steps, knockback, and spawn recovery.  Horizontal slab
# gaps therefore need 40u of motion margin above the standing hull.
MIN_SAFE_HEADROOM = 96
MIN_SANDWICH_OVERLAP = PLAYER_DIAMETER + 16
SPAWN_LINK_LIFT = 9       # engine link raises authored DM origins before CM probes
SPAWN_COLUMN_SWEEP = MIN_SAFE_HEADROOM - PLAYER_H
# Atlas proves 96u total standing clearance by sweeping the linked 56u hull
# upward 40u.  Integer source brushes must begin strictly above the final hull
# top, so the reserved floor-to-overhead interval is 24+9+32+40+1 = 106u.
SPAWN_COMPILED_COLUMN_HEIGHT = (
    -PLAYER_MINS_Z + SPAWN_LINK_LIFT + PLAYER_MAXS_Z
    + SPAWN_COLUMN_SWEEP + 1
)
SPAWN_ESCAPE_DISTANCE = 96
SPAWN_ESCAPE_STEP = 16
JUMP_H      = 50           # max jump height (units)
HOOK_MIN    = 150          # minimum useful hook distance
HOOK_MAX    = 580          # maximum hook range
HOOK_EYE_Z  = 22           # hook trace starts at player origin + view height
HOOK_RELEASE_TICK_MSEC = 100
HOOK_RELEASE_TICKS = tuple(range(1, 7))
MAX_HOOK_SOURCES_PER_GEOMETRY = 2
MAX_HOOK_CANDIDATES_V4 = 512
HOOK_RELEASES_PER_SOURCE = 4        # two ranks x four schedules x 64 cells
HOOK_COORDINATE_GEOMETRY_FLOOR = 43  # complete former 512/(2*6) prefix
MAX_RUNTIME_HOOK_ZONES = 256  # must match ml_bridge.c MAX_HOOK_ZONES
DM_SPAWN_COUNT = 8         # headroom for six-player live matches
MIN_SPAWN_SEPARATION = 384 # minimum 2D spacing for generated DM spawns
SPAWN_EDGE_MARGIN = 96     # keep starts away from room edges/sky walls
SPAWN_SOLID_MARGIN = 48    # extra XY clearance from cover and low blockers
LIGHT_REGION_SIZE = 512    # maximum XY extent of one base-floor light region
LIGHT_SAMPLE_SPACING = 128 # fixed coverage probes within each region
MIN_FLOOR_LIGHT_COVERAGE = 0.98
FLOOR_LIGHT_RADIUS = 448   # guaranteed horizontal reach used by validation
SPAWN_LIGHT_EYE_OFFSET = SPAWN_LINK_LIFT + 22
SPAWN_COLUMN_LIGHT_OFFSET = 48  # 17u above the admitted spawn eye
OVERHEAD_LIGHT_VALUE = 900 # qrad point-light intensity
UNDER_PLATFORM_LIGHT_VALUE = 700
MIN_FLOOR_LIGHT_VALUE = 650
INTERIOR_LIGHT_VALUE = 850
INTERIOR_LIGHT_RADIUS = 384
MIN_INTERIOR_LIGHT_VALUE = 800
MIN_INTERIOR_LIGHT_RADIUS = 320
WORLD_AMBIENT_LIGHT = 180
KILL_PLANE_DROP = 64       # distance below the lowest playable floor
KILL_PLANE_MARGIN = 512    # XY overhang beyond the generated layout
KILL_PLANE_DAMAGE = 100000 # instant-kill damage for out-of-bounds falls
KILL_PLANE_SPAWNFLAGS = 12 # trigger_hurt: SILENT | NO_PROTECTION
KILL_PLANE_SURFACE_VALUE = 31337 # hook-impact marker; must match g_local.h
LETHAL_GUARD_HEIGHT = 96   # higher than a normal jump; contains void edges
LETHAL_GUARD_THICKNESS = WALL_T
# Keep source-side lethal-edge witnesses identical to the compiled challenge:
# first prefer the midpoint, then try a bounded deterministic spread along the
# exact segment.  The inward offset clears the 16u guard and 16u hull radius by
# one Pmove fixed-point quantum.
LETHAL_EDGE_SAMPLE_FRACTIONS = (
    0.5, 0.1, 0.9, 0.25, 0.75, 0.05, 0.95, 0.33, 0.67, 0.4, 0.6,
)
LETHAL_EDGE_INWARD_OFFSET = (
    LETHAL_GUARD_THICKNESS + PLAYER_XY_HALF + PMOVE_FIXED_QUANTUM
)

# Hook zone flags — must match ml_bridge.h
HOOK_CEILING  = 1
HOOK_WALL     = 2
HOOK_REQUIRED = 4

# ── Textures ─────────────────────────────────────────────────────────────────

T_FLOOR  = "e1u1/floor1_3"
T_CEIL   = "e1u1/ceil1_1"
T_WALL   = "e1u1/wswall1_1"
T_TRIM   = "e1u1/flat1_2"
T_LIGHT  = "e1u1/baselt_2"
T_METAL  = "e1u1/metal1_1"
T_LAVA   = "e1u1/brlava"
T_SKY    = "unit1_"          # built-in sky environment prefix
T_SKY_SURFACE = "e1u1/sky1"  # visible sky brush texture
SURF_SKY = 0x4               # q_shared.h: do not draw, add to skybox
SURF_NODRAW = 0x80           # q_shared.h: no rendered surface reference
SURF_LIGHT = 0x1             # q_shared.h: emit light from this surface
SURF_WARP  = 0x8             # q_shared.h: turbulent (liquid) surface
CONTENTS_SOLID = 1           # q_shared.h: solid brush contents
CONTENTS_LAVA  = 8           # q_shared.h: lava contents (engine applies damage)

# All names verified present in this install's pak0.pak
PALETTES = {
    "base":  dict(floor="e1u1/floor1_3", ceil="e1u1/ceil1_1",
                  wall="e1u1/wswall1_1", trim="e1u1/flat1_2",
                  metal="e1u1/metal1_1", light="e1u1/baselt_2"),
    "metal": dict(floor="e1u1/floor3_1", ceil="e1u1/ceil1_3",
                  wall="e1u1/metal1_2",  trim="e1u1/flat1_1",
                  metal="e1u1/metal1_4", light="e1u1/baselt_3"),
    "rock":  dict(floor="e1u1/floor3_3", ceil="e1u1/ceil1_4",
                  wall="e1u1/rocks19_1", trim="e1u1/flat1_2",
                  metal="e1u1/metal1_3", light="e1u1/baselt_5"),
}

# ── Terrain / structure constants ─────────────────────────────────────────────

TERRACE_STEP = 96          # elevation per terrace level (> JUMP_H: needs stairs)
STAIR_STEP_H = 16          # riser height
STAIR_STEP_D = 32          # tread depth
STAIR_WIDTH  = 128
TOWER_BASE   = 128         # tower footprint (square)
TOWER_H_MIN, TOWER_H_MAX = 192, 288   # hook-required (> JUMP_H), within HOOK_MAX
LANE_WALL_H  = 128         # tall enough to block jumps → must flank
LANE_WALL_T  = 32
LANE_GAP     = 192         # central gap in lane walls
# Keep each lane-wall endpoint on the arena-center side of the complete
# protected spawn-clearance hull.  This is the same geometry that defines the
# ring's inner boundary in _normalize_spawn_arena_floor, so the anchor arena
# retains its defining lanes without weakening any certified spawn witness.
LANE_EDGE_INSET = SPAWN_EDGE_MARGIN + PLAYER_XY_HALF + SPAWN_SOLID_MARGIN
LAVA_DEPTH   = 8
LAVA_MIN, LAVA_MAX = 224, 384   # pool edge length

ARENA_STYLES = ("arena_open", "arena_vertical", "arena_lanes")
STYLES = ("open", "towers", "canyon", "pits", *ARENA_STYLES, "mixed")

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float; y: float; z: float
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def length(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def as_tuple(self): return (self.x, self.y, self.z)


@dataclass
class Room:
    gx: int; gy: int           # grid position
    wx: int; wy: int           # world origin (min corner)
    w:  int; d:  int           # width, depth (world units)
    floor_z: int               # floor surface Z
    ceil_z:  int               # ceiling surface Z
    kind: str                  # 'arena'|'room'|'corridor'
    platforms: List[dict] = field(default_factory=list)  # {z, x0,y0,x1,y1}


@dataclass
class Connection:
    a: int; b: int             # room indices
    kind: str                  # 'door'|'drop'|'shaft'
    # world-space opening centre
    cx: int; cy: int; cz: int
    width: int; height: int


@dataclass
class HookZone:
    anchor:  Tuple[float,float,float]
    landing: Tuple[float,float,float]
    distance: float
    flags: int


@dataclass(frozen=True)
class HookClaimCandidateV4:
    """Unproven source-bound hook schedule with a desired landing proposal."""

    source: Tuple[float, float, float]
    trace_target: Tuple[float, float, float]
    landing: Tuple[float, float, float]
    release_after_ticks: int
    distance_milliunits: int
    flags: int


@dataclass
class SolidBox:
    x0: int; y0: int; z0: int
    x1: int; y1: int; z1: int


@dataclass(frozen=True)
class HorizontalSurface:
    """Thin overhead/platform/roof assembly used by the trap-gap contract."""

    surface_id: str
    kind: str
    box: SolidBox


@dataclass(frozen=True)
class InteriorLightZone:
    """One enterable room-like location that owns an internal light."""

    zone_id: str
    kind: str
    bounds: Tuple[int, int, int, int]
    floor_z: int
    ceiling_z: int
    anchor: Tuple[float, float, float]


@dataclass(frozen=True)
class FloorLightRegion:
    """A fixed, measurable patch of spawn-clear base-floor space."""

    region_id: str
    bounds: Tuple[int, int, int, int]
    floor_z: int
    samples: Tuple[Tuple[float, float, float], ...]


@dataclass(frozen=True)
class LightSource:
    """A generated point light participating in the floor-light contract."""

    region_id: str
    kind: str
    origin: Tuple[float, float, float]
    radius: float
    value: int


def _xy_overlap_size(a: SolidBox, b: SolidBox) -> Tuple[int, int]:
    return (
        max(0, min(a.x1, b.x1) - max(a.x0, b.x0)),
        max(0, min(a.y1, b.y1) - max(a.y0, b.y0)),
    )


def horizontal_sandwich_gap(
    a: HorizontalSurface,
    b: HorizontalSurface,
) -> Optional[int]:
    """Return an unsafe free gap, or ``None`` when the pair is acceptable.

    A pair is player-admitting only when its overlapping footprint is at least
    48x48u (the 32u hull plus 8u lateral margin on each side).  Free vertical
    gaps from the 56u standing hull height through 95u are rejected; 96u is the
    minimum safe movement headroom.  Touching/merged slabs are always safe.
    """
    overlap_x, overlap_y = _xy_overlap_size(a.box, b.box)
    if (overlap_x < MIN_SANDWICH_OVERLAP or
            overlap_y < MIN_SANDWICH_OVERLAP):
        return None
    lower, upper = ((a.box, b.box) if a.box.z0 <= b.box.z0
                    else (b.box, a.box))
    gap = upper.z0 - lower.z1
    if PLAYER_H <= gap < MIN_SAFE_HEADROOM:
        return gap
    return None


def unsafe_horizontal_sandwiches(
    surfaces: List[HorizontalSurface],
) -> List[Tuple[HorizontalSurface, HorizontalSurface, int]]:
    """Deterministically enumerate all player-admitting trap gaps."""
    ordered = sorted(
        surfaces,
        key=lambda item: (item.box.z0, item.box.z1, item.surface_id),
    )
    unsafe = []
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            gap = horizontal_sandwich_gap(first, second)
            if gap is not None:
                unsafe.append((first, second, gap))
    return unsafe


def light_reaches_sample(source: LightSource,
                         sample: Tuple[float, float, float],
                         occluders: List[SolidBox]) -> bool:
    """Return whether a point light has a direct path to a floor sample.

    Coverage intentionally uses horizontal reach rather than pretending to
    duplicate qrad's build-specific falloff.  The emitted ``light`` value
    controls brightness; this contract guarantees that a sufficiently strong
    source is nearby and that platforms/roofs do not sit between it and the
    base floor.
    """
    sx, sy, sz = sample
    lx, ly, lz = source.origin
    if lz <= sz or math.hypot(lx - sx, ly - sy) > source.radius:
        return False
    height = lz - sz
    for box in occluders:
        if not (sz < box.z0 < lz):
            continue
        t = (box.z0 - sz) / height
        ix = sx + (lx - sx) * t
        iy = sy + (ly - sy) * t
        if box.x0 <= ix <= box.x1 and box.y0 <= iy <= box.y1:
            return False
    return True


def _segment_intersects_solid(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    box: SolidBox,
) -> bool:
    """Return whether a closed segment intersects an axis-aligned solid."""
    entry = 0.0
    exit_ = 1.0
    for start_value, end_value, lower, upper in zip(
        start, end,
        (box.x0, box.y0, box.z0),
        (box.x1, box.y1, box.z1),
    ):
        delta = end_value - start_value
        if abs(delta) <= 1e-12:
            if start_value < lower or start_value > upper:
                return False
            continue
        first = (lower - start_value) / delta
        second = (upper - start_value) / delta
        if first > second:
            first, second = second, first
        entry = max(entry, first)
        exit_ = min(exit_, second)
        if entry > exit_:
            return False
    return True


def light_reaches_spawn_eye(
    source: LightSource,
    eye: Tuple[float, float, float],
    solids: Sequence[SolidBox],
) -> bool:
    """Mirror the compiled spawn-light obligation in source coordinates.

    Unlike floor coverage, this uses full three-dimensional radius and every
    emitted solid. The compiled CM trace remains the promotion authority.
    """
    if math.dist(eye, source.origin) > source.radius:
        return False
    return not any(
        _segment_intersects_solid(eye, source.origin, box)
        for box in solids
    )


def floor_light_coverage(region: FloorLightRegion,
                         sources: List[LightSource],
                         occluders: List[SolidBox]) -> float:
    """Fraction of a region's spawn-clear samples with direct point light."""
    if not region.samples:
        return 0.0
    lit = sum(
        any(light_reaches_sample(source, sample, occluders)
            for source in sources)
        for sample in region.samples
    )
    return lit / len(region.samples)


# ── Heat-field placement engine ───────────────────────────────────────────────
#
# Multi-channel 3D heat field over the world. Items carry placement templates:
# an EMISSION (heat they deposit when placed) and a PULL vector (attraction to
# / repulsion from each channel's accumulated heat). Placement runs coarse →
# fine in cubic reduction steps; every placement deposits heat that steers all
# later placements, so the item economy self-organises: weapons spread out,
# counters answer threats (armour gravitates toward railgun heat), supplies
# trail engagement space, and nothing crowds spawns.

HEAT_CHANNELS = ("weapon", "defense", "supply", "danger", "spawn", "objective")


class HeatField:
    """Sparse multi-channel heat lattice with per-level cell sizes."""

    def __init__(self, cell: int = 256):
        self.cell = cell
        self.data: dict = {}   # (channel, ix, iy, iz) -> float

    def _key(self, ch, x, y, z):
        c = self.cell
        return (ch, int(x // c), int(y // c), int(z // c))

    def deposit(self, ch: str, x: float, y: float, z: float,
                amount: float, radius: float):
        """Deposit heat with linear falloff out to radius."""
        c = self.cell
        r_cells = max(1, int(radius // c))
        ix, iy, iz = int(x // c), int(y // c), int(z // c)
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-max(1, r_cells // 2), max(1, r_cells // 2) + 1):
                    d = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if d > r_cells:
                        continue
                    falloff = 1.0 - (d / (r_cells + 1))
                    k = (ch, ix + dx, iy + dy, iz + dz)
                    self.data[k] = self.data.get(k, 0.0) + amount * falloff

    def sample(self, ch: str, x: float, y: float, z: float) -> float:
        return self.data.get(self._key(ch, x, y, z), 0.0)

    def score(self, pulls: dict, x: float, y: float, z: float) -> float:
        return sum(wgt * self.sample(ch, x, y, z) for ch, wgt in pulls.items())


# Placement templates. emit: {channel: amount}; radius: emission reach;
# pull: {channel: weight} — positive seeks heat, negative avoids it.
ITEM_TEMPLATES = {
    # Tier 1 (coarse pass): power weapons define the map economy.
    "weapon_railgun":        dict(tier=1, radius=700,
                                  emit={"weapon": 1.2},
                                  pull={"weapon": -1.7, "spawn": -0.6, "objective": 0.4}),
    "weapon_rocketlauncher": dict(tier=1, radius=700,
                                  emit={"weapon": 1.2},
                                  pull={"weapon": -1.7, "spawn": -0.6, "objective": 0.3}),
    # Tier 2: counters answer tier-1 heat. Armour lives where weapon heat is
    # high — if the enemy grabs the rail, your answer is nearby, not with it.
    "item_armor_body":       dict(tier=2, radius=500,
                                  emit={"defense": 1.0},
                                  pull={"weapon": 0.6, "defense": -1.2, "spawn": -0.3}),
    "item_armor_combat":     dict(tier=2, radius=450,
                                  emit={"defense": 0.7},
                                  pull={"weapon": 0.7, "defense": -1.0}),
    "weapon_supershotgun":   dict(tier=2, radius=500,
                                  emit={"weapon": 0.7},
                                  pull={"weapon": -1.0, "spawn": 0.2}),
    "weapon_chaingun":       dict(tier=2, radius=500,
                                  emit={"weapon": 0.7},
                                  pull={"weapon": -1.0, "spawn": 0.2}),
    "weapon_grenadelauncher": dict(tier=2, radius=500,
                                  emit={"weapon": 0.7},
                                  pull={"weapon": -0.9, "danger": 0.3}),
    "weapon_hyperblaster":   dict(tier=2, radius=500,
                                  emit={"weapon": 0.7},
                                  pull={"weapon": -1.0}),
    # Runes (persistent, placed for ML training). Offense runes gravitate to
    # weapon heat (gear up where the fights are); survival runes to defense
    # heat (answer threats); vampire bridges. Mutually repel via "objective"
    # so they spread. One per type per map (see _place_runes).
    "rune_strength":         dict(tier=2, radius=450, rune=True,
                                  emit={"weapon": 0.5, "objective": 0.4},
                                  pull={"weapon": 0.9, "objective": -1.0, "spawn": -0.4}),
    "rune_haste":            dict(tier=2, radius=450, rune=True,
                                  emit={"weapon": 0.4, "objective": 0.4},
                                  pull={"weapon": 0.7, "objective": -1.0, "spawn": -0.4}),
    "rune_regen":            dict(tier=2, radius=450, rune=True,
                                  emit={"defense": 0.5, "objective": 0.4},
                                  pull={"defense": 0.8, "objective": -1.0, "spawn": -0.3}),
    "rune_vampire":          dict(tier=2, radius=450, rune=True,
                                  emit={"weapon": 0.3, "defense": 0.3, "objective": 0.4},
                                  pull={"weapon": 0.6, "defense": 0.4, "objective": -1.0}),
    "rune_resist":           dict(tier=2, radius=450, rune=True,
                                  emit={"defense": 0.5, "objective": 0.4},
                                  pull={"defense": 0.7, "objective": -1.0, "spawn": -0.3}),
    # Tier 3 (fine pass): supplies trail the fight space created above.
    "item_health_large":     dict(tier=3, radius=350,
                                  emit={"supply": 0.6},
                                  pull={"weapon": 0.5, "supply": -1.0, "danger": 0.2}),
    "item_health":           dict(tier=3, radius=300,
                                  emit={"supply": 0.4},
                                  pull={"defense": 0.4, "supply": -0.8}),
    "item_adrenaline":       dict(tier=3, radius=300,
                                  emit={"supply": 0.5},
                                  pull={"danger": 0.5, "supply": -0.8, "spawn": -0.4}),
    "ammo_rockets":          dict(tier=3, radius=250,
                                  emit={"supply": 0.3},
                                  pull={"weapon": 1.1, "supply": -0.6}),
    "ammo_slugs":            dict(tier=3, radius=250,
                                  emit={"supply": 0.3},
                                  pull={"weapon": 1.1, "supply": -0.6}),
    "ammo_cells":            dict(tier=3, radius=250,
                                  emit={"supply": 0.3},
                                  pull={"weapon": 0.9, "supply": -0.6}),
    "ammo_grenades":         dict(tier=3, radius=250,
                                  emit={"supply": 0.3},
                                  pull={"weapon": 0.6, "supply": -0.6}),
    "ammo_shells":           dict(tier=3, radius=250,
                                  emit={"supply": 0.3},
                                  pull={"weapon": 0.6, "supply": -0.6}),
    "ammo_bullets":          dict(tier=3, radius=250,
                                  emit={"supply": 0.3},
                                  pull={"weapon": 0.6, "supply": -0.6}),
    # Tier 0: structure-only items. Never floor-placed by the tier passes,
    # but they emit so the economy reacts to them (mega heat repels supplies).
    "item_health_mega":      dict(tier=0, radius=450,
                                  emit={"supply": 0.9, "objective": 0.3},
                                  pull={}),
}

# Structure loot is chosen FROM these pools BY the heat field (judge finding:
# hardcoded weapon+armour pairs collapsed the counter economy to 32u and
# oversupplied armour 19:12 against power weapons). Tower tops gate high
# value; platforms take the broader mid-tier table.
TOWER_LOOT_POOL = [
    "weapon_railgun", "weapon_rocketlauncher",
    "item_armor_body", "item_armor_combat", "item_health_large",
]
PLATFORM_LOOT_POOL = TOWER_LOOT_POOL + [
    "weapon_hyperblaster", "weapon_chaingun", "weapon_supershotgun",
    "weapon_grenadelauncher", "item_adrenaline",
]
POWER_WEAPONS = ("weapon_railgun", "weapon_rocketlauncher")
ARMOR_CLASSES = ("item_armor_body", "item_armor_combat")
POWER_MIN_SEPARATION = 600   # judge: spreads of 269-422u = clumped control points
ARMOR_MIN_FROM_POWER = 350   # judge: counters answer just outside melee, never on top
WEAPON_CLASS_CAP = 3         # judge: 7 chainguns collapsed weapon diversity
SPAWN_ROUTE_ANCHOR_CLASSES = (
    "weapon_supershotgun", "weapon_chaingun",
    "item_armor_combat", "item_health_large",
)

# Demand-driven ammo pairing: each weapon class placed creates ammo demand
# (~1 box per 2 weapons of that class) in the tier-3 pass.
AMMO_FOR_WEAPON = {
    "weapon_railgun":         "ammo_slugs",
    "weapon_rocketlauncher":  "ammo_rockets",
    "weapon_hyperblaster":    "ammo_cells",
    "weapon_chaingun":        "ammo_bullets",
    "weapon_supershotgun":    "ammo_shells",
    "weapon_grenadelauncher": "ammo_grenades",
}

# Per-tier lattice resolution: each pass re-reads the previous pass's heat at
# a finer cell size — the "cubic space reduction" schedule.
TIER_CELL = {1: 512, 2: 256, 3: 128}


# ── Brush writer ──────────────────────────────────────────────────────────────

class MapWriter:
    def __init__(self):
        self.brushes: List[str] = []
        self.entities: List[str] = []
        self.world_props: dict = {}
        self.solid_boxes: List[SolidBox] = []

    # ----- primitive: axis-aligned box brush (6 faces) -----

    def _face(self, p1, p2, p3, tex, xo=0, yo=0, rot=0, xs=1, ys=1,
              surf_flags=0, contents=0, value=0):
        # Q2 .map format: texture xoff yoff rot xscale yscale contents surf_flags value
        return (f"( {p1[0]} {p1[1]} {p1[2]} ) "
                f"( {p2[0]} {p2[1]} {p2[2]} ) "
                f"( {p3[0]} {p3[1]} {p3[2]} ) "
                f"{tex} {xo} {yo} {rot} {xs} {ys} {contents} {surf_flags} {value}")

    def make_box_brush(self, x0,y0,z0, x1,y1,z1,
                       tf=None, tc=None, tw=None,
                       surf_flags=0, contents=0, value=0) -> str:
        tf = tf or T_FLOOR
        tc = tc or T_CEIL
        tw = tw or T_WALL
        f = self._face
        # q2tools computes normal as (p0-p1)×(p2-p1) — windings below produce
        # outward-facing normals (away from solid, into empty space) as q2bsp expects.
        faces = [
            f((x1,y0,z0),(x1,y1,z0),(x0,y1,z0), tf, surf_flags=surf_flags, contents=contents, value=value),  # bottom -Z
            f((x1,y1,z1),(x1,y0,z1),(x0,y0,z1), tc, surf_flags=surf_flags, contents=contents, value=value),  # top    +Z
            f((x1,y0,z1),(x1,y0,z0),(x0,y0,z0), tw, surf_flags=surf_flags, contents=contents, value=value),  # south  -Y
            f((x0,y1,z0),(x1,y1,z0),(x1,y1,z1), tw, surf_flags=surf_flags, contents=contents, value=value),  # north  +Y
            f((x0,y0,z0),(x0,y1,z0),(x0,y1,z1), tw, surf_flags=surf_flags, contents=contents, value=value),  # west   -X
            f((x1,y1,z0),(x1,y0,z0),(x1,y0,z1), tw, surf_flags=surf_flags, contents=contents, value=value),  # east   +X
        ]
        return "{\n" + "\n".join(faces) + "\n}"

    def add_brush(self, x0,y0,z0, x1,y1,z1,
                  tf=None, tc=None, tw=None,
                  surf_flags=0, contents=0, value=0):
        self.brushes.append(self.make_box_brush(
            x0, y0, z0, x1, y1, z1,
            tf=tf, tc=tc, tw=tw,
            surf_flags=surf_flags, contents=contents, value=value,
        ))
        if not (contents & CONTENTS_LAVA):
            self.solid_boxes.append(SolidBox(x0, y0, z0, x1, y1, z1))

    def add_entity(self, classname: str, props: dict):
        lines = ['{', f'"classname" "{classname}"']
        for k, v in props.items():
            lines.append(f'"{k}" "{v}"')
        lines.append('}')
        self.entities.append("\n".join(lines))

    def add_brush_entity(self, classname: str, props: dict, brushes: List[str]):
        lines = ['{', f'"classname" "{classname}"']
        for k, v in props.items():
            lines.append(f'"{k}" "{v}"')
        lines.extend(brushes)
        lines.append('}')
        self.entities.append("\n".join(lines))

    def write(self, path: Path):
        lines = ['// Generated by q2-ml-bot/maps/generator.py',
                 '// Game: Quake II', '']
        # worldspawn
        lines.append('{')
        lines.append('"classname" "worldspawn"')
        lines.append('"message" "ML Training Map"')
        lines.append(f'"sky" "{T_SKY}"')
        lines.append(f'"light" "{WORLD_AMBIENT_LIGHT}"')
        for key, value in self.world_props.items():
            lines.append(f'"{key}" "{value}"')
        for b in self.brushes:
            lines.append(b)
        lines.append('}')
        lines.append('')
        for e in self.entities:
            lines.append(e)
            lines.append('')
        path.write_text("\n".join(lines))


# ── Room generator ────────────────────────────────────────────────────────────

class MapGenerator:
    def __init__(self, seed: Optional[int] = None, style: str = "mixed",
                 gym: bool = False):
        self.rng = random.Random(seed)
        self.gym = bool(gym)   # rune gym: static-place runes for controlled teaching
        self.rooms: List[Room] = []
        self.connections: List[Connection] = []
        self.hook_zones: List[HookZone] = []
        self.hook_claim_candidates_v4: List[Tuple[str, HookClaimCandidateV4]] = []
        self.spawn_points: List[Tuple[int, int, int]] = []
        self.spawn_blockers: List[SolidBox] = []
        # Four connected standing-volume bands reserve a deterministic arena
        # perimeter before any platform or emitted structure is planned.
        self._spawn_protected_domains: List[SolidBox] = []
        self._spawn_protected_witnesses: List[Tuple[int, int, int]] = []
        self._spawn_protected_anchor: Optional[Room] = None
        self.light_occluders: List[SolidBox] = []
        self.light_regions: List[FloorLightRegion] = []
        self.light_sources: List[LightSource] = []
        self.light_coverages: dict = {}
        self.spawn_eye_samples: List[Tuple[float, float, float]] = []
        self.horizontal_surfaces: List[HorizontalSurface] = []
        self.lethal_edges: List[dict] = []
        self.lethal_guard_walls: List[SolidBox] = []
        self.interior_light_zones: List[InteriorLightZone] = []
        self.interior_light_sources: List[LightSource] = []
        self._interior_zone_specs: List[
            Tuple[str, str, Tuple[int, int, int, int], int, int]
        ] = []
        self.writer = MapWriter()
        self._adjacent: List[Tuple[int, int]] = []   # grid-neighbour room pairs
        self.lava_pools: List[SolidBox] = []
        self.objectives: List[dict] = []             # lattice-seeded value sites
        self._placed_loot: List[Tuple[str, int, int, int]] = []  # heat emitters
        self._heat_placed: List[Tuple[str, int, int, int]] = []
        # One physical origin may publish at most one item entity.  Keep this
        # reservation set authoritative through objectives, authored rewards,
        # heat placement, and relaxation so later passes cannot stack items.
        self._item_origins: set[Tuple[int, int, int]] = set()
        self._loot_sites: List[Tuple[str, int, int, int]] = []  # (kind, x, y, z)
        self._observed_heat: List[dict] = []   # play-telemetry deposits
        self.relax_moves = 0
        self.tower_count = 0
        self.stair_count = 0
        self.lane_wall_count = 0
        self.cover_count = 0
        self.hallway_count = 0
        self.corner_count = 0
        self.large_building_count = 0

        # Resolve style → feature knobs (mixed rolls per map)
        r = self.rng
        if style == "mixed":
            style = r.choice(STYLES[:-1])
        self.style = style
        self.palette_name = r.choice(list(PALETTES.keys()))
        self.pal = PALETTES[self.palette_name]
        self.terrace_levels = {
            "open": 1, "towers": 2, "canyon": 1, "pits": 3,
            "arena_open": 1, "arena_vertical": 3, "arena_lanes": 1,
        }[style]
        self.tower_prob = {
            "open": 0.15, "towers": 0.9, "canyon": 0.2, "pits": 0.3,
            "arena_open": 0.05, "arena_vertical": 0.75, "arena_lanes": 0.15,
        }[style]
        # Judge: zero lane walls leaves verticality carrying all the flow —
        # every style keeps some chance of engineered chokepoints.
        self.lane_prob = {
            "open": 0.35, "towers": 0.35, "canyon": 0.9, "pits": 0.3,
            "arena_open": 0.15, "arena_vertical": 0.30, "arena_lanes": 1.0,
        }[style]
        self.lava_prob = {
            "open": 0.1, "towers": 0.15, "canyon": 0.15, "pits": 0.6,
            "arena_open": 0.05, "arena_vertical": 0.10, "arena_lanes": 0.05,
        }[style]
        # Arena presets bias the graph toward repeated combat bowls while
        # retaining ordinary rooms for circulation, pickups, and respawns.
        self.occupied_density = 0.55
        self.extra_arena_prob = 0.0
        self.corridor_prob = 0.35
        self.arena_cover_range = (2, 4)
        self.hallway_ratio = 0.0
        self.corner_range = (1, 1)
        self.large_building_ratio = 0.0
        if style == "arena_open":
            self.occupied_density = 0.48
            self.extra_arena_prob = 0.45
            self.corridor_prob = 0.18
            self.arena_cover_range = (4, 7)
            self.hallway_ratio = 0.18
            self.corner_range = (2, 3)
            self.large_building_ratio = 0.25
        elif style == "arena_vertical":
            self.occupied_density = 0.52
            self.extra_arena_prob = 0.40
            self.corridor_prob = 0.20
            self.arena_cover_range = (3, 5)
            self.hallway_ratio = 0.18
            self.corner_range = (1, 2)
            self.large_building_ratio = 0.25
        elif style == "arena_lanes":
            self.occupied_density = 0.52
            self.extra_arena_prob = 0.50
            self.corridor_prob = 0.18
            self.arena_cover_range = (3, 6)
            self.hallway_ratio = 0.18
            self.corner_range = (2, 3)
            self.large_building_ratio = 0.30

    # ----- grid layout -----

    def _room_params(self, kind: str):
        r = self.rng
        if kind == 'arena':
            w = 3 * GRID_SIZE
            d = 3 * GRID_SIZE
            h = r.randint(384, 512)
        elif kind == 'room':
            w = r.randint(1, 2) * GRID_SIZE
            d = r.randint(1, 2) * GRID_SIZE
            h = r.randint(256, 384)
        else:  # corridor
            # True through-hallways: one cell wide, two cells long, with a
            # deliberately lower ceiling than rooms and arenas.
            if r.random() < 0.5:
                w, d = 2 * GRID_SIZE, GRID_SIZE
            else:
                w, d = GRID_SIZE, 2 * GRID_SIZE
            h = r.randint(160, 224)
        return w, d, h

    def _normalize_room_ceiling_sandwiches(self):
        """Separate unsafe overlapping ceiling bands monotonically.

        Room footprints deliberately overlap to make one connected arena.  A
        96u terrace offset can leave a second, reachable ceiling plate only
        56--95u above the first.  Moving a lower band down and later merging it
        upward can cycle between the same room states.  Instead, keep the
        lower authored band fixed and raise only the upper room until the
        exact pair has 96u of free headroom. Every edit is a strict integer
        ceiling increase, consumes no RNG, and explicit height/iteration
        fences fail closed if a future surface model violates that invariant.
        """
        initial_max_ceiling = max(
            (room.ceil_z for room in self.rooms), default=0
        )
        ceiling_limit = initial_max_ceiling + max(1, len(self.rooms)) * 2 * (
            MIN_SAFE_HEADROOM + WALL_T + 4
        )
        max_iterations = sum(
            ceiling_limit - room.ceil_z for room in self.rooms
        ) + 1
        seen_states = set()
        for _iteration in range(max_iterations):
            state = tuple(room.ceil_z for room in self.rooms)
            if state in seen_states:
                raise RuntimeError(
                    "room overhead normalization repeated a ceiling state"
                )
            seen_states.add(state)
            surfaces = self._room_overhead_surfaces()
            unsafe = unsafe_horizontal_sandwiches(surfaces)
            if not unsafe:
                return
            first, second, _gap = unsafe[0]
            lower_surface, upper_surface = (
                (first, second)
                if first.box.z0 <= second.box.z0
                else (second, first)
            )
            lower_index = int(lower_surface.surface_id.rsplit("_", 1)[1])
            upper_index = int(upper_surface.surface_id.rsplit("_", 1)[1])
            if lower_index == upper_index:
                raise RuntimeError("room overhead assembly contains an unsafe gap")
            upper_room = self.rooms[upper_index]
            required_upper_bottom = (
                lower_surface.box.z1 + MIN_SAFE_HEADROOM
            )
            increase = int(required_upper_bottom - upper_surface.box.z0)
            if increase <= 0:
                raise RuntimeError(
                    "room overhead normalization did not require an increase"
                )
            prior_ceiling = upper_room.ceil_z
            upper_room.ceil_z += increase
            if upper_room.ceil_z <= prior_ceiling:
                raise RuntimeError(
                    "room overhead normalization was not monotonic"
                )
            if upper_room.ceil_z > ceiling_limit:
                raise RuntimeError(
                    "room overhead normalization exceeded its ceiling bound"
                )
        raise RuntimeError(
            "room overhead normalization exceeded its finite iteration budget"
        )

    def _room_overhead_surfaces(self) -> List[HorizontalSurface]:
        surfaces = []
        for index, room in enumerate(self.rooms):
            surfaces.append(HorizontalSurface(
                f"room_ceiling_{index}", "ceiling",
                SolidBox(room.wx, room.wy, room.ceil_z,
                         room.wx + room.w, room.wy + room.d,
                         room.ceil_z + WALL_T),
            ))
            if room.kind == "arena":
                lx = room.wx + room.w // 4
                ly = room.wy + room.d // 4
                surfaces.append(HorizontalSurface(
                    f"room_light_panel_{index}", "light_panel",
                    SolidBox(lx, ly, room.ceil_z - 4,
                             lx + room.w // 2, ly + room.d // 2,
                             room.ceil_z + WALL_T),
                ))
        return surfaces

    def _register_room_overheads(self):
        self.horizontal_surfaces.extend(self._room_overhead_surfaces())

    @staticmethod
    def _room_footprints_overlap(first: Room, second: Room) -> bool:
        return (
            first.wx < second.wx + second.w
            and first.wx + first.w > second.wx
            and first.wy < second.wy + second.d
            and first.wy + first.d > second.wy
        )

    def _normalize_spawn_arena_floor(self) -> None:
        """Keep one arena perimeter available as a standing component.

        Overlapping terrace plates are intentional, but a higher ordinary
        room can otherwise cut every large arena into a long, narrow exposed
        strip.  Then hundreds of individually legal starts exist while no one
        source component can satisfy the unchanged 1024-by-1024 span gate.

        Select the largest, highest deterministic arena and lower only higher
        floors that overlap its footprint, preserving each affected room's
        authored height.  A lower room ceiling that intersects the anchor's
        96-unit standing column is raised just to the safe boundary.  This is
        a pre-geometry layout invariant: it consumes no RNG, creates no rescue
        path, and leaves clearance, separation, span, and component checks
        untouched.
        """
        arenas = [room for room in self.rooms if room.kind == "arena"]
        if not arenas:
            self._spawn_protected_domains = []
            self._spawn_protected_witnesses = []
            self._spawn_protected_anchor = None
            return
        anchor = max(
            arenas,
            key=lambda room: (
                room.w * room.d,
                room.floor_z,
                -room.wy,
                -room.wx,
            ),
        )
        self._spawn_protected_anchor = anchor
        # spawn_blockers conservatively model ceiling/light assemblies from
        # ceil_z-8, so raise the authored ceiling until that lower bound is at
        # the first integer-safe compiled-column boundary.
        safe_ceiling = anchor.floor_z + SPAWN_COMPILED_COLUMN_HEIGHT + 8
        for room in self.rooms:
            if room is anchor or not self._room_footprints_overlap(anchor, room):
                continue
            if room.floor_z > anchor.floor_z:
                delta = room.floor_z - anchor.floor_z
                room.floor_z -= delta
                room.ceil_z -= delta
            ceiling_bottom = room.ceil_z - 8
            ceiling_top = room.ceil_z + WALL_T
            standing_bottom = anchor.floor_z + 1
            standing_top = anchor.floor_z + SPAWN_COMPILED_COLUMN_HEIGHT
            if ceiling_top > standing_bottom and ceiling_bottom < standing_top:
                room.ceil_z = max(room.ceil_z, safe_ceiling)

        # Candidate centers begin SPAWN_EDGE_MARGIN units inside the room.
        # Protect their full generous spawn-clearance hull along a connected
        # perimeter ring.  The four bands overlap at the corners, provide more
        # than 1024 units of span on both axes, and leave the arena center free
        # for ordinary cover/objective geometry.
        spawn_half = PLAYER_XY_HALF + SPAWN_SOLID_MARGIN
        outer_inset = SPAWN_EDGE_MARGIN - spawn_half
        inner_inset = SPAWN_EDGE_MARGIN + spawn_half
        z0 = anchor.floor_z + 1
        z1 = anchor.floor_z + SPAWN_COMPILED_COLUMN_HEIGHT
        x0, x1 = anchor.wx + outer_inset, anchor.wx + anchor.w - outer_inset
        y0, y1 = anchor.wy + outer_inset, anchor.wy + anchor.d - outer_inset
        inner_x0 = anchor.wx + inner_inset
        inner_x1 = anchor.wx + anchor.w - inner_inset
        inner_y0 = anchor.wy + inner_inset
        inner_y1 = anchor.wy + anchor.d - inner_inset
        self._spawn_protected_domains = [
            SolidBox(x0, y0, z0, inner_x0, y1, z1),
            SolidBox(inner_x1, y0, z0, x1, y1, z1),
            SolidBox(x0, y0, z0, x1, inner_y0, z1),
            SolidBox(x0, inner_y1, z0, x1, y1, z1),
        ]

        low_x = anchor.wx + SPAWN_EDGE_MARGIN
        high_x = anchor.wx + anchor.w - SPAWN_EDGE_MARGIN
        low_y = anchor.wy + SPAWN_EDGE_MARGIN
        high_y = anchor.wy + anchor.d - SPAWN_EDGE_MARGIN
        mid_x = low_x + ((high_x - low_x) // 128) * 64
        mid_y = low_y + ((high_y - low_y) // 128) * 64
        origin_z = anchor.floor_z + 24
        self._spawn_protected_witnesses = [
            (low_x, low_y, origin_z),
            (mid_x, low_y, origin_z),
            (high_x, low_y, origin_z),
            (high_x, mid_y, origin_z),
            (high_x, high_y, origin_z),
            (mid_x, high_y, origin_z),
            (low_x, high_y, origin_z),
            (low_x, mid_y, origin_z),
        ]

    def _overlaps_spawn_protected_domain(self, box: SolidBox) -> bool:
        return any(
            self._boxes_overlap(box, protected)
            for protected in self._spawn_protected_domains
        )

    def _assert_spawn_protected_domain_clear(self) -> None:
        """Fail closed if any final source obstacle enters the reservation."""
        for kind, obstacles in (
            ("spawn blocker", self.spawn_blockers),
            ("lava pool", self.lava_pools),
        ):
            for index, obstacle in enumerate(obstacles):
                if self._overlaps_spawn_protected_domain(obstacle):
                    raise RuntimeError(
                        f"{kind} {index} overlaps the protected spawn domain"
                    )

    def _assert_spawn_protected_capacity(self) -> None:
        """Prove the reserved ring still owns an exact admissible spawn set."""
        self._assert_spawn_protected_domain_clear()
        if not self._spawn_protected_witnesses:
            return
        anchor = self._spawn_protected_anchor
        witnesses = self._spawn_protected_witnesses
        if anchor is None or len(witnesses) != DM_SPAWN_COUNT:
            raise RuntimeError("protected spawn domain lacks eight witnesses")
        if len(set(witnesses)) != DM_SPAWN_COUNT:
            raise RuntimeError("protected spawn witnesses are not unique")
        if not self._spawn_span_ok(witnesses):
            raise RuntimeError("protected spawn witnesses lack map span")
        if min(
            math.hypot(left[0] - right[0], left[1] - right[1])
            for index, left in enumerate(witnesses)
            for right in witnesses[index + 1:]
        ) < MIN_SPAWN_SEPARATION:
            raise RuntimeError("protected spawn witnesses lack separation")
        if not all(
            self._spawn_is_locally_clear(anchor, x, y, z)
            for x, y, z in witnesses
        ):
            raise RuntimeError("protected spawn witness is not locally clear")

        from maps.routes import source_endpoint_components

        components = source_endpoint_components(
            self.rooms, witnesses, self.spawn_blockers, self.lava_pools,
        )
        if (
            len(components) != DM_SPAWN_COUNT
            or len(set(components.values())) != 1
        ):
            raise RuntimeError(
                "protected spawn witnesses do not share one source component"
            )

    def build_layout(self, grid_n: int = 5):
        """Generate room graph using Prim's MST + extra edges on grid_n × grid_n."""
        rng = self.rng
        N = grid_n

        # 1. Decide which cells have rooms (at least 40% density)
        occupied: set = set()
        cell_level: dict = {}
        # Start with a random seed cell; levels random-walk outward so
        # terraces are spatially coherent (neighbouring cells differ by ≤1).
        start = (rng.randint(1, N-2), rng.randint(1, N-2))
        frontier = [start]
        occupied.add(start)
        cell_level[start] = 0
        target_rooms = max(8, int(N * N * self.occupied_density))
        while frontier and len(occupied) < target_rooms:
            c = rng.choice(frontier)
            nx = c[0] + rng.choice([-1, 0, 1])
            ny = c[1] + rng.choice([-1, 0, 0, 1])
            nc = (nx, ny)
            if 0 <= nx < N and 0 <= ny < N and nc not in occupied:
                occupied.add(nc)
                frontier.append(nc)
                step = rng.choice([-1, 0, 0, 1]) if self.terrace_levels > 1 else 0
                cell_level[nc] = max(0, min(self.terrace_levels - 1,
                                            cell_level[c] + step))

        # 2. Assign room kinds — more central = arena, edges = room/corridor
        grid_rooms: dict = {}  # (gx,gy) -> Room index
        for (gx, gy) in occupied:
            dist_centre = abs(gx - N//2) + abs(gy - N//2)
            if (dist_centre <= 1 and len(occupied) > 5) or (
                rng.random() < self.extra_arena_prob
            ):
                kind = 'arena'
            elif rng.random() < self.corridor_prob:
                kind = 'corridor'
            else:
                kind = 'room'

            w, d, h = self._room_params(kind)
            floor_z = cell_level.get((gx, gy), 0) * TERRACE_STEP
            room = Room(gx=gx, gy=gy,
                        wx=gx*GRID_SIZE, wy=gy*GRID_SIZE,
                        w=w, d=d, floor_z=floor_z, ceil_z=floor_z+h,
                        kind=kind)
            grid_rooms[(gx, gy)] = len(self.rooms)
            self.rooms.append(room)

        if self.style in ARENA_STYLES:
            # Enforce the composition instead of trusting independent random
            # rolls: at least two low-ceiling hallways and one mid-ceiling
            # ordinary room balance the repeated high-ceiling arenas.
            target_hallways = max(2, int(round(len(self.rooms) * self.hallway_ratio)))
            corridors = [room for room in self.rooms if room.kind == 'corridor']
            candidates = sorted(
                (room for room in self.rooms if room.kind != 'corridor'),
                key=lambda room: abs(room.gx - N // 2) + abs(room.gy - N // 2),
                reverse=True,
            )
            for room in candidates[:max(0, target_hallways - len(corridors))]:
                room.kind = 'corridor'
                room.w, room.d, height = self._room_params('corridor')
                room.ceil_z = room.floor_z + height
            if not any(room.kind == 'room' for room in self.rooms):
                candidates = sorted(
                    (room for room in self.rooms if room.kind == 'arena'),
                    key=lambda room: abs(room.gx - N // 2) + abs(room.gy - N // 2),
                    reverse=True,
                )
                if candidates:
                    room = candidates[0]
                    room.kind = 'room'
                    room.w, room.d, height = self._room_params('room')
                    room.ceil_z = room.floor_z + height

        # Establish one map-spanning standing-floor domain before connections,
        # platforms, and static blockers are derived from the room geometry.
        self._normalize_spawn_arena_floor()

        # 3. Connect adjacent rooms
        dirs = [(1,0),(0,1),(-1,0),(0,-1)]
        for (gx, gy), idx_a in grid_rooms.items():
            for dx, dy in dirs:
                nb = (gx+dx, gy+dy)
                if nb in grid_rooms:
                    idx_b = grid_rooms[nb]
                    if idx_a < idx_b:
                        self._make_connection(idx_a, idx_b)
                        self._adjacent.append((idx_a, idx_b))

        # 4. Overlapping room footprints must not create reachable low voids
        # between separate ceiling/light bands.
        self._normalize_room_ceiling_sandwiches()
        self._register_room_overheads()

        # 5. Add platforms to arenas and tall rooms. Candidate platforms are
        # checked against all registered overhead assemblies.
        for room_index, room in enumerate(self.rooms):
            if room.kind in ('arena', 'room'):
                self._add_platforms(room, room_index)

        # 6. Register every floor plate as a spawn blocker: with terracing,
        # a higher room's plate can overhang a lower room's area, and spawns
        # or items placed there would be inside solid rock. Boundary contact
        # (standing ON a plate) does not count as overlap.
        for room in self.rooms:
            self._admit_spawn_blocker(SolidBox(
                room.wx, room.wy, -WALL_T,
                room.wx + room.w, room.wy + room.d, room.floor_z),
                required=True,
            )
            # Ceilings too: a low corridor ceiling can overhang a higher
            # neighbouring terrace at head height. (-8 also covers light strips.)
            self._admit_spawn_blocker(SolidBox(
                room.wx, room.wy, room.ceil_z - 8,
                room.wx + room.w, room.wy + room.d, room.ceil_z + WALL_T),
                required=True,
            )

        # The skybox/kill plane is a final containment fallback, not playable
        # traversal.  Seal every edge of the union of room floor plates so a
        # normal movement or knockback path cannot enter that death volume.
        self._plan_lethal_drop_guards()
        self._assert_spawn_protected_capacity()

    def _plan_lethal_drop_guards(self) -> None:
        """Build guard walls around playable floor-union edges facing void."""
        self.lethal_edges = []
        self.lethal_guard_walls = []
        if not self.rooms:
            return

        xs = sorted({value for room in self.rooms
                     for value in (room.wx, room.wx + room.w)})
        ys = sorted({value for room in self.rooms
                     for value in (room.wy, room.wy + room.d)})
        covered: dict[Tuple[int, int], int] = {}
        for ix, (x0, x1) in enumerate(zip(xs, xs[1:])):
            x = (x0 + x1) / 2.0
            for iy, (y0, y1) in enumerate(zip(ys, ys[1:])):
                y = (y0 + y1) / 2.0
                floors = [
                    room.floor_z for room in self.rooms
                    if room.wx <= x < room.wx + room.w
                    and room.wy <= y < room.wy + room.d
                ]
                if floors:
                    covered[(ix, iy)] = max(floors)

        directions = (
            ("west", -1, 0), ("east", 1, 0),
            ("south", 0, -1), ("north", 0, 1),
        )
        thickness = LETHAL_GUARD_THICKNESS
        planned: List[Tuple[dict, SolidBox]] = []
        for (ix, iy), floor_z in sorted(covered.items()):
            x0, x1 = xs[ix], xs[ix + 1]
            y0, y1 = ys[iy], ys[iy + 1]
            for side, dx, dy in directions:
                if (ix + dx, iy + dy) in covered:
                    continue
                if side == "west":
                    wall = SolidBox(x0, y0, floor_z,
                                    x0 + thickness, y1,
                                    floor_z + LETHAL_GUARD_HEIGHT)
                    edge = [x0, y0, x0, y1, floor_z]
                elif side == "east":
                    wall = SolidBox(x1 - thickness, y0, floor_z,
                                    x1, y1,
                                    floor_z + LETHAL_GUARD_HEIGHT)
                    edge = [x1, y0, x1, y1, floor_z]
                elif side == "south":
                    wall = SolidBox(x0, y0, floor_z,
                                    x1, y0 + thickness,
                                    floor_z + LETHAL_GUARD_HEIGHT)
                    edge = [x0, y0, x1, y0, floor_z]
                else:
                    wall = SolidBox(x0, y1 - thickness, floor_z,
                                    x1, y1,
                                    floor_z + LETHAL_GUARD_HEIGHT)
                    edge = [x0, y1, x1, y1, floor_z]
                if not self._lethal_edge_has_standing_witness(side, edge):
                    continue
                planned.append(({"side": side, "segment": edge}, wall))

        # Do not let an earlier planned guard influence a later source-geometry
        # witness.  All claims are challenged against the same pre-guard solid
        # set, then published together.
        for edge, wall in planned:
            self.lethal_edges.append(edge)
            self.lethal_guard_walls.append(wall)
            self._admit_spawn_blocker(wall, required=True)
            self.light_occluders.append(wall)

    def _lethal_edge_has_standing_witness(self, side: str,
                                          segment: List[int]) -> bool:
        """Return whether the edge has one deterministic safe interior stand.

        Floor-union construction selects the highest overlapping room floor.
        A lower room's ceiling or a platform can nevertheless occupy the
        standing column above that floor.  Publishing such a location as a
        playable lethal edge makes the compiled claim probe start solid.
        """
        outward = {
            "west": (-1.0, 0.0), "east": (1.0, 0.0),
            "south": (0.0, -1.0), "north": (0.0, 1.0),
        }[side]
        x0, y0, x1, y1, floor_z = segment
        for fraction in LETHAL_EDGE_SAMPLE_FRACTIONS:
            edge_x = x0 + (x1 - x0) * fraction
            edge_y = y0 + (y1 - y0) * fraction
            interior_x = edge_x - outward[0] * LETHAL_EDGE_INWARD_OFFSET
            interior_y = edge_y - outward[1] * LETHAL_EDGE_INWARD_OFFSET
            if self._player_column_is_clear(interior_x, interior_y, floor_z):
                return True
        return False

    def _make_connection(self, ia: int, ib: int):
        a, b = self.rooms[ia], self.rooms[ib]
        # Find shared wall direction
        if a.gx == b.gx:   # vertical neighbours (y axis)
            cx = a.wx + min(a.w, b.w) // 2
            cy = max(a.wy, b.wy)
            cz = max(a.floor_z, b.floor_z)
            kind = 'door' if abs(a.floor_z - b.floor_z) < JUMP_H else 'drop'
            w = min(128, min(a.w, b.w) // 2)
        else:               # horizontal neighbours (x axis)
            cx = max(a.wx, b.wx)
            cy = a.wy + min(a.d, b.d) // 2
            cz = max(a.floor_z, b.floor_z)
            kind = 'door' if abs(a.floor_z - b.floor_z) < JUMP_H else 'drop'
            w = min(128, min(a.d, b.d) // 2)

        height = 112 if kind == 'door' else 64  # drops are low openings
        conn = Connection(a=ia, b=ib, kind=kind,
                          cx=cx, cy=cy, cz=cz, width=w, height=height)
        self.connections.append(conn)

    def _add_platforms(self, room: Room, room_index: int):
        """Add floating platforms with at least 96u overhead headroom."""
        rng = self.rng
        room_h = room.ceil_z - room.floor_z
        if room_h < 256:
            return
        n_platforms = rng.randint(1, 2) if room.kind == 'arena' else 1
        max_platform_z = room.ceil_z - MIN_SAFE_HEADROOM - 20
        min_platform_z = room.floor_z + 128
        if max_platform_z < min_platform_z:
            return
        for platform_index in range(n_platforms):
            accepted = None
            for _attempt in range(16):
                plat_z = rng.randint(min_platform_z, max_platform_z)
                plat_w = rng.randint(96, min(256, room.w - 64))
                plat_d = rng.randint(96, min(256, room.d - 64))
                plat_x0 = room.wx + rng.randint(
                    32, max(33, room.w - plat_w - 32)
                )
                plat_y0 = room.wy + rng.randint(
                    32, max(33, room.d - plat_d - 32)
                )
                surface = HorizontalSurface(
                    f"platform_{room_index}_{platform_index}", "platform",
                    # Include the 8u underside trim and 16u platform slab as
                    # one physical assembly for free-gap measurement.
                    SolidBox(plat_x0, plat_y0, plat_z - 8,
                             plat_x0 + plat_w, plat_y0 + plat_d, plat_z + 16),
                )
                if any(horizontal_sandwich_gap(surface, existing) is not None
                       for existing in self.horizontal_surfaces):
                    continue
                if not self._admit_spawn_blocker(
                    surface.box, register=False
                ):
                    continue
                accepted = (plat_z, plat_w, plat_d, plat_x0, plat_y0, surface)
                break
            if accepted is None:
                continue
            plat_z, plat_w, plat_d, plat_x0, plat_y0, surface = accepted
            self._admit_spawn_blocker(surface.box, required=True)
            room.platforms.append({
                'z': plat_z, 'thick': 16,
                'x0': plat_x0, 'y0': plat_y0,
                'x1': plat_x0 + plat_w, 'y1': plat_y0 + plat_d,
            })
            self.horizontal_surfaces.append(surface)
            # With terraces, a platform from one room can overhang another
            # room's floor at body height — keep spawns/items out of it.
            # (Covers the railing strip at z-8 too.)
            self.light_occluders.append(SolidBox(
                plat_x0, plat_y0, plat_z - 8,
                plat_x0 + plat_w, plat_y0 + plat_d, plat_z + 16))

    # ----- brush generation -----

    def _emit_room(self, room: Room):
        w = self.writer
        pal = self.pal
        x0, y0  = room.wx, room.wy
        x1, y1  = x0 + room.w, y0 + room.d
        fz, cz  = room.floor_z, room.ceil_z

        # Floor plate (drops to the lowest terrace so cliff faces are sealed)
        plate_bottom = -WALL_T
        w.add_brush(x0, y0, plate_bottom,  x1, y1, fz,
                    tf=pal['floor'], tc=pal['floor'], tw=pal['wall'])
        # Ceiling
        w.add_brush(x0, y0, cz,  x1, y1, cz + WALL_T,
                    tf=pal['ceil'], tc=pal['ceil'], tw=pal['ceil'])
        # Internal perimeter walls are intentionally omitted. The skybox seals
        # the level, while overlapping floor plates create one connected combat
        # space. Cover blocks below provide evasion without trapping bots.
        self._emit_cover(room)

        # Platforms
        for p in room.platforms:
            w.add_brush(p['x0'], p['y0'], p['z'],
                        p['x1'], p['y1'], p['z'] + p['thick'],
                        tf=pal['metal'], tc=pal['metal'], tw=pal['trim'])
            # Railing / trim strip below platform
            w.add_brush(p['x0'], p['y0'], p['z'] - 8,
                        p['x1'], p['y1'], p['z'],
                        tf=pal['trim'], tc=pal['trim'], tw=pal['trim'])

        # Ceiling light strips in arenas (SURF_LIGHT so the radiosity stage
        # emits from them when the light pass is enabled)
        if room.kind == 'arena':
            lx = x0 + room.w // 4
            ly = y0 + room.d // 4
            lw = room.w // 2
            ld = room.d // 2
            w.add_brush(lx, ly, cz - 4,  lx+lw, ly+ld, cz,
                        tf=pal['light'], tc=pal['light'], tw=pal['trim'],
                        surf_flags=SURF_LIGHT, value=300)

    def _emit_connection(self, conn: Connection):
        """Connections are graph metadata only in the open-layout generator."""
        _ = conn
        return

    def _emit_cover(self, room: Room):
        """Emit waist-high cover that creates engagement/evasion choices."""
        if room.w < 640 or room.d < 640:
            return
        w = self.writer
        fz = room.floor_z
        cx = room.wx + room.w // 2
        cy = room.wy + room.d // 2
        cover_h = 96

        if room.kind == 'arena':
            blocks = [
                (cx - 320, cy - 64, cx - 192, cy + 192),
                (cx + 192, cy - 192, cx + 320, cy + 64),
            ]
        else:
            blocks = [
                (cx - 64, cy - 160, cx + 64, cy + 160),
            ]

        for x0, y0, x1, y1 in blocks:
            box = SolidBox(x0, y0, fz, x1, y1, fz + cover_h)
            if not self._admit_spawn_blocker(box):
                continue
            w.add_brush(x0, y0, fz, x1, y1, fz + cover_h,
                        tf=self.pal['metal'], tc=self.pal['metal'], tw=self.pal['trim'])

    @staticmethod
    def _boxes_overlap(a: SolidBox, b: SolidBox) -> bool:
        return not (
            a.x1 <= b.x0 or a.x0 >= b.x1 or
            a.y1 <= b.y0 or a.y0 >= b.y1 or
            a.z1 <= b.z0 or a.z0 >= b.z1
        )

    def _admit_spawn_blocker(
        self, box: SolidBox, *, register: bool = True,
        required: bool = False,
    ) -> bool:
        """Centrally admit a standing solid before any brush is emitted.

        Optional geometry receives ``False`` when it would enter the protected
        spawn ring.  Required shell geometry fails closed.  This is the only
        method allowed to append to ``spawn_blockers``.
        """
        if self._overlaps_spawn_protected_domain(box):
            if required:
                raise RuntimeError(
                    "required standing solid overlaps the protected spawn domain"
                )
            return False
        if register:
            self.spawn_blockers.append(box)
        return True

    def _structure_is_clear(self, box: SolidBox) -> bool:
        if not self._admit_spawn_blocker(box, register=False):
            return False
        if any(self._boxes_overlap(box, blocker)
               for blocker in self.spawn_blockers):
            return False
        # Hollow buildings and corner pockets promise traversable interior
        # volume, not just solid boundary brushes. Later towers, cover, lava,
        # and objectives must challenge that volume as part of ordinary
        # structure placement or they can silently erase a promised zone.
        if self._overlaps_reserved_interior(box):
            return False
        # A late structure must not invalidate an already-emitted reward.
        # Reserve the exact standing hull around each item origin in addition
        # to its zero-volume point identity.
        return not any(
            self._boxes_overlap(
                box,
                SolidBox(
                    x - PLAYER_XY_HALF, y - PLAYER_XY_HALF,
                    z + PLAYER_MINS_Z,
                    x + PLAYER_XY_HALF, y + PLAYER_XY_HALF,
                    z + PLAYER_MAXS_Z,
                ),
            )
            for x, y, z in self._item_origins
        )

    def _overlaps_reserved_interior(self, box: SolidBox) -> bool:
        """Protect every promised interior's traversable light volume."""
        for _zone_id, _kind, bounds, floor_z, ceiling_z in self._interior_zone_specs:
            x0, y0, x1, y1 = bounds
            protected = SolidBox(
                x0, y0, floor_z, x1, y1,
                max(ceiling_z, floor_z + MIN_SAFE_HEADROOM),
            )
            if self._boxes_overlap(box, protected):
                return True
        return False

    def _emit_hallways(self):
        """Turn corridor rooms into recognizable low-ceiling through-halls.

        Parallel full-height side walls define a 256u clear passage while
        both ends remain open. The room's existing ceiling supplies the low
        overhead band, so hallways are useful circulation rather than traps.
        """
        for room in self.rooms:
            if room.kind != 'corridor':
                continue
            # The low-ceiling corridor room remains an authored hallway even
            # when its optional side-wall assembly is suppressed to preserve
            # the protected spawn ring.
            self.hallway_count += 1
            fz = room.floor_z
            top = room.ceil_z
            thickness = 24
            passage_half = 128
            walls: List[SolidBox] = []
            if room.w >= room.d:  # passage runs east/west
                cy = room.wy + room.d // 2
                walls = [
                    SolidBox(room.wx + 32, cy - passage_half - thickness, fz,
                             room.wx + room.w - 32, cy - passage_half, top),
                    SolidBox(room.wx + 32, cy + passage_half, fz,
                             room.wx + room.w - 32, cy + passage_half + thickness, top),
                ]
            else:                 # passage runs north/south
                cx = room.wx + room.w // 2
                walls = [
                    SolidBox(cx - passage_half - thickness, room.wy + 32, fz,
                             cx - passage_half, room.wy + room.d - 32, top),
                    SolidBox(cx + passage_half, room.wy + 32, fz,
                             cx + passage_half + thickness, room.wy + room.d - 32, top),
                ]
            if not all(
                self._admit_spawn_blocker(box, register=False)
                for box in walls
            ):
                continue
            for box in walls:
                self._admit_spawn_blocker(box, required=True)
                self.writer.add_brush(
                    box.x0, box.y0, box.z0, box.x1, box.y1, box.z1,
                    tf=self.pal['wall'], tc=self.pal['ceil'], tw=self.pal['wall'],
                )

    def _emit_corner_pockets(self):
        """Place L-shaped cover pockets at arena edges for real corner play."""
        rng = self.rng
        for room in self.rooms:
            if room.kind != 'arena':
                continue
            corners = [
                (room.wx + 192, room.wy + 192, 1, 1),
                (room.wx + room.w - 192, room.wy + 192, -1, 1),
                (room.wx + 192, room.wy + room.d - 192, 1, -1),
                (room.wx + room.w - 192, room.wy + room.d - 192, -1, -1),
            ]
            rng.shuffle(corners)
            target = rng.randint(*self.corner_range)
            placed = 0
            for x, y, sx, sy in corners:
                if placed >= target:
                    break
                length, thick, height = 144, 24, 112
                horizontal = SolidBox(
                    min(x, x + sx * length), y - thick // 2, room.floor_z,
                    max(x, x + sx * length), y + thick // 2,
                    room.floor_z + height,
                )
                vertical = SolidBox(
                    x - thick // 2, min(y, y + sy * length), room.floor_z,
                    x + thick // 2, max(y, y + sy * length),
                    room.floor_z + height,
                )
                if not (self._structure_is_clear(horizontal) and
                        self._structure_is_clear(vertical)):
                    continue
                pocket_bounds = (
                    min(x, x + sx * length) + thick,
                    min(y, y + sy * length) + thick,
                    max(x, x + sx * length) - thick,
                    max(y, y + sy * length) - thick,
                )
                pocket_x = (pocket_bounds[0] + pocket_bounds[2]) / 2.0
                pocket_y = (pocket_bounds[1] + pocket_bounds[3]) / 2.0
                if not self._player_column_is_clear(
                        pocket_x, pocket_y, room.floor_z):
                    continue
                for box in (horizontal, vertical):
                    self._admit_spawn_blocker(box, required=True)
                    self.writer.add_brush(
                        box.x0, box.y0, box.z0, box.x1, box.y1, box.z1,
                        tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['wall'],
                    )
                self._interior_zone_specs.append((
                    f"corner_{self.corner_count}", "corner_pocket",
                    pocket_bounds, room.floor_z, room.ceil_z,
                ))
                self.corner_count += 1
                placed += 1

    def _emit_large_buildings(self):
        """Add enterable roofed buildings to arena presets.

        Each structure is hollow, has 144u doors on opposite faces, and adds
        both an interior low ceiling and an optional roof control point.
        """
        if self.large_building_ratio <= 0.0:
            return
        rng = self.rng
        arenas = [room for room in self.rooms if room.kind == 'arena']
        if self.style == "arena_vertical":
            # Highest terraces cannot be buried by another room's floor plate.
            arenas.sort(key=lambda room: (room.floor_z, room.w * room.d), reverse=True)
        else:
            rng.shuffle(arenas)
        target = max(1, int(round(len(arenas) * self.large_building_ratio)))
        for room in arenas:
            if self.large_building_count >= target:
                break
            size = rng.choice((384, 448))
            # Vertical layouts stack room volumes. A deliberately lower
            # building roof keeps the structure below neighbouring ceiling
            # plates while still creating a distinct indoor ceiling band.
            height = (rng.randint(144, 160) if self.style == "arena_vertical"
                      else rng.randint(176, 224))
            inset = 96
            candidates = [
                (room.wx + inset, room.wy + inset),
                (room.wx + room.w - size - inset, room.wy + inset),
                (room.wx + inset, room.wy + room.d - size - inset),
                (room.wx + room.w - size - inset,
                 room.wy + room.d - size - inset),
                (room.wx + (room.w - size) // 2, room.wy + inset),
                (room.wx + (room.w - size) // 2,
                 room.wy + room.d - size - inset),
                (room.wx + inset, room.wy + (room.d - size) // 2),
                (room.wx + room.w - size - inset,
                 room.wy + (room.d - size) // 2),
            ]
            rng.shuffle(candidates)
            for x0, y0 in candidates:
                fz = room.floor_z
                shell = SolidBox(x0, y0, fz, x0 + size, y0 + size, fz + height + 24)
                if not self._structure_is_clear(shell):
                    continue
                thick, door = 24, 144
                door0 = x0 + (size - door) // 2
                door1 = door0 + door
                walls = [
                    SolidBox(x0, y0, fz, x0 + thick, y0 + size, fz + height),
                    SolidBox(x0 + size - thick, y0, fz,
                             x0 + size, y0 + size, fz + height),
                    SolidBox(x0, y0, fz, door0, y0 + thick, fz + height),
                    SolidBox(door1, y0, fz, x0 + size, y0 + thick, fz + height),
                    SolidBox(x0, y0 + size - thick, fz,
                             door0, y0 + size, fz + height),
                    SolidBox(door1, y0 + size - thick, fz,
                             x0 + size, y0 + size, fz + height),
                ]
                roof = SolidBox(x0, y0, fz + height,
                                x0 + size, y0 + size, fz + height + 24)
                roof_surface = HorizontalSurface(
                    f"building_roof_{self.large_building_count}", "roof", roof
                )
                if any(horizontal_sandwich_gap(roof_surface, existing) is not None
                       for existing in self.horizontal_surfaces):
                    continue
                for box in (*walls, roof):
                    self._admit_spawn_blocker(box, required=True)
                    self.writer.add_brush(
                        box.x0, box.y0, box.z0, box.x1, box.y1, box.z1,
                        tf=self.pal['metal'], tc=self.pal['ceil'], tw=self.pal['wall'],
                    )
                self.light_occluders.append(roof)
                self.horizontal_surfaces.append(roof_surface)
                cx, cy = x0 + size // 2, y0 + size // 2
                self._interior_zone_specs.append((
                    f"building_{self.large_building_count}", "building",
                    (x0 + thick, y0 + thick,
                     x0 + size - thick, y0 + size - thick),
                    fz, fz + height,
                ))
                self._loot_sites.append(("platform", cx, cy, fz + 24))
                self._loot_sites.append(("tower", cx, cy, fz + height + 48))
                self.large_building_count += 1
                break

    # ----- v2 structures: stairs, towers, lane walls, lava -----

    def _emit_stairs(self):
        """Stair runs across grid boundaries where adjacent terraces differ by
        more than jump height — elevation becomes walkable, not just hookable."""
        w = self.writer
        for ia, ib in self._adjacent:
            a, b = self.rooms[ia], self.rooms[ib]
            if a.floor_z == b.floor_z:
                continue
            low, high = (a, b) if a.floor_z < b.floor_z else (b, a)
            dz = high.floor_z - low.floor_z
            if dz <= JUMP_H:
                continue
            n = (dz + STAIR_STEP_H - 1) // STAIR_STEP_H
            steps: List[SolidBox] = []

            if a.gx != b.gx:    # east/west neighbours: stairs run along X
                bx = max(a.gx, b.gx) * GRID_SIZE      # shared cell boundary
                y0 = max(a.wy, b.wy)
                y1 = min(a.wy + a.d, b.wy + b.d)
                if y1 - y0 < STAIR_WIDTH + 32:
                    continue
                cy = (y0 + y1) // 2
                run_dir = -1 if low.gx < high.gx else 1   # steps extend into the low room
                for i in range(1, n + 1):
                    step_top = low.floor_z + i * STAIR_STEP_H
                    sx1 = bx + run_dir * (n - i) * STAIR_STEP_D
                    sx0 = bx + run_dir * (n - i + 1) * STAIR_STEP_D
                    x0, x1 = min(sx0, sx1), max(sx0, sx1)
                    steps.append(SolidBox(
                        x0, cy - STAIR_WIDTH // 2, low.floor_z,
                        x1, cy + STAIR_WIDTH // 2, step_top,
                    ))
            else:               # north/south neighbours: stairs run along Y
                by = max(a.gy, b.gy) * GRID_SIZE
                x0r = max(a.wx, b.wx)
                x1r = min(a.wx + a.w, b.wx + b.w)
                if x1r - x0r < STAIR_WIDTH + 32:
                    continue
                cx = (x0r + x1r) // 2
                run_dir = -1 if low.gy < high.gy else 1
                for i in range(1, n + 1):
                    step_top = low.floor_z + i * STAIR_STEP_H
                    sy1 = by + run_dir * (n - i) * STAIR_STEP_D
                    sy0 = by + run_dir * (n - i + 1) * STAIR_STEP_D
                    y0, y1 = min(sy0, sy1), max(sy0, sy1)
                    steps.append(SolidBox(
                        cx - STAIR_WIDTH // 2, y0, low.floor_z,
                        cx + STAIR_WIDTH // 2, y1, step_top,
                    ))

            # A partial staircase is neither safe traversal nor truthful map
            # metadata.  Reserve or emit the complete run atomically.
            if not steps or any(
                not self._admit_spawn_blocker(step, register=False)
                for step in steps
            ):
                continue
            for step in steps:
                self._admit_spawn_blocker(step, required=True)
                w.add_brush(
                    step.x0, step.y0, step.z0,
                    step.x1, step.y1, step.z1,
                    tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['trim'],
                )
            self.stair_count += 1

    def _place_objectives(self):
        """Place one lattice-seeded objective after static geometry settles.

        The objective still owns its guard geometry, but both the tower and
        pickup are selected against the complete blocker set.  This keeps the
        exported opportunity prior honest: no later cover pass can turn the
        promised pickup into a startsolid endpoint."""
        rng = self.rng
        w = self.writer
        arenas = sorted((r for r in self.rooms if r.kind == 'arena'),
                        key=lambda r: r.w * r.d, reverse=True)
        if not arenas:
            return
        chosen = None
        for room in arenas:
            for _ in range(24):
                th = rng.randint(TOWER_H_MIN + 32, TOWER_H_MAX)
                tx = room.wx + rng.randint(192, max(193, room.w - TOWER_BASE - 192))
                ty = room.wy + rng.randint(192, max(193, room.d - TOWER_BASE - 192))
                fz = room.floor_z
                top = min(fz + th, room.ceil_z - PLAYER_H - 24)
                if top - fz < TOWER_H_MIN + 32:
                    continue
                box = SolidBox(tx, ty, fz,
                               tx + TOWER_BASE, ty + TOWER_BASE, top)
                surface = HorizontalSurface(
                    "objective_tower", "tower", box
                )
                if (
                    not self._structure_is_clear(box)
                    or any(
                        horizontal_sandwich_gap(surface, existing) is not None
                        for existing in self.horizontal_surfaces
                    )
                ):
                    continue
                self._admit_spawn_blocker(box, required=True)
                quad_site = self._floor_item_spot(
                    room, tx + TOWER_BASE // 2, ty + TOWER_BASE // 2,
                    TOWER_BASE + 128,
                )
                self.spawn_blockers.pop()
                if quad_site is not None:
                    chosen = (
                        room, tx, ty, fz, top, box, surface, quad_site
                    )
                    break
            if chosen is not None:
                break
        if chosen is None:
            return
        room, tx, ty, fz, top, box, surface, quad_site = chosen
        self._admit_spawn_blocker(box, required=True)
        w.add_brush(tx, ty, fz, tx + TOWER_BASE, ty + TOWER_BASE, top,
                    tf=self.pal['light'], tc=self.pal['metal'], tw=self.pal['metal'])
        self.horizontal_surfaces.append(surface)
        quad_x, quad_y, quad_z = quad_site
        self._reserve_item_origin(quad_site, "item_quad objective")
        w.add_entity("item_quad", {"origin": f"{quad_x} {quad_y} {quad_z}"})
        self.objectives.append({
            "item": "item_quad", "x": quad_x, "y": quad_y, "z": quad_z,
            "guard": "tower_base", "height": top - fz, "value": 1.0,
        })
        self.tower_count += 1

    def _emit_towers(self):
        """Hook-target towers in arenas: high-value loot on top, reachable only
        by grapple (or platform chains). Registers hook zones."""
        rng = self.rng
        w = self.writer
        for room in self.rooms:
            if room.kind != 'arena' or rng.random() > self.tower_prob:
                continue
            for _ in range(rng.randint(1, 2)):
                th = rng.randint(TOWER_H_MIN, TOWER_H_MAX)
                tx = room.wx + rng.randint(160, max(161, room.w - TOWER_BASE - 160))
                ty = room.wy + rng.randint(160, max(161, room.d - TOWER_BASE - 160))
                fz = room.floor_z
                top = fz + th
                if top > room.ceil_z - PLAYER_H - 24:
                    top = room.ceil_z - PLAYER_H - 24
                    th = top - fz
                if th < TOWER_H_MIN:
                    continue
                box = SolidBox(tx, ty, fz,
                               tx + TOWER_BASE, ty + TOWER_BASE, top)
                surface = HorizontalSurface(
                    f"tower_{self.tower_count}", "tower", box
                )
                if (
                    not self._structure_is_clear(box)
                    or any(
                        horizontal_sandwich_gap(surface, existing) is not None
                        for existing in self.horizontal_surfaces
                    )
                ):
                    continue
                self._admit_spawn_blocker(box, required=True)
                w.add_brush(tx, ty, fz, tx + TOWER_BASE, ty + TOWER_BASE, top,
                            tf=self.pal['metal'], tc=self.pal['metal'], tw=self.pal['wall'])
                self.horizontal_surfaces.append(surface)
                cx_t = tx + TOWER_BASE // 2
                cy_t = ty + TOWER_BASE // 2
                # Tower-top loot site: the heat pass decides WHAT goes here —
                # weapon repulsion diversifies nearby towers instead of every
                # top getting an identical power-weapon + armour pair.
                self._loot_sites.append(("tower", cx_t, cy_t, top + 24))
                self.tower_count += 1

    def _emit_lane_walls(self):
        """Parallel sight-blocking walls with central gaps in arenas: creates
        lanes, chokepoints, and flanking routes (too tall to jump)."""
        rng = self.rng
        w = self.writer
        for room in self.rooms:
            if room.kind != 'arena' or rng.random() > self.lane_prob:
                continue
            fz = room.floor_z
            cx = room.wx + room.w // 2
            cy = room.wy + room.d // 2
            along_x = rng.random() < 0.5
            segments: List[SolidBox] = []
            for side in (-1, 1):
                if along_x:
                    wy = cy + side * room.d // 4
                    seg_x0 = room.wx + LANE_EDGE_INSET
                    seg_x1 = room.wx + room.w - LANE_EDGE_INSET
                    gap0 = cx - LANE_GAP // 2
                    gap1 = cx + LANE_GAP // 2
                    for x0, x1 in ((seg_x0, gap0), (gap1, seg_x1)):
                        if x1 - x0 < 64:
                            continue
                        segments.append(SolidBox(
                            x0, wy - LANE_WALL_T // 2, fz,
                            x1, wy + LANE_WALL_T // 2, fz + LANE_WALL_H,
                        ))
                else:
                    wx_ = cx + side * room.w // 4
                    seg_y0 = room.wy + LANE_EDGE_INSET
                    seg_y1 = room.wy + room.d - LANE_EDGE_INSET
                    gap0 = cy - LANE_GAP // 2
                    gap1 = cy + LANE_GAP // 2
                    for y0, y1 in ((seg_y0, gap0), (gap1, seg_y1)):
                        if y1 - y0 < 64:
                            continue
                        segments.append(SolidBox(
                            wx_ - LANE_WALL_T // 2, y0, fz,
                            wx_ + LANE_WALL_T // 2, y1, fz + LANE_WALL_H,
                        ))
            if len(segments) != 4 or any(
                not self._admit_spawn_blocker(segment, register=False)
                for segment in segments
            ):
                continue
            for segment in segments:
                self._admit_spawn_blocker(segment, required=True)
                w.add_brush(
                    segment.x0, segment.y0, segment.z0,
                    segment.x1, segment.y1, segment.z1,
                    tf=self.pal['wall'], tc=self.pal['trim'], tw=self.pal['wall'],
                )
            # Historical count is one per parallel wall, each split into two
            # brush segments around the central gap.
            self.lane_wall_count += 2

    def _emit_lava_pools(self):
        """Sunken lava hazards with mega-health beside them: risk/reward
        terrain, and honest danger ground-truth for the voxel lattice.
        Real CONTENTS_LAVA — the engine applies damage natively."""
        rng = self.rng
        w = self.writer
        spawn_room = self._combat_spawn_room()
        for room in self.rooms:
            if room is spawn_room or room.kind == 'corridor':
                continue
            if min(room.w, room.d) < 640 or rng.random() > self.lava_prob:
                continue
            fz = room.floor_z
            placement = None
            for _ in range(24):
                size = rng.randint(LAVA_MIN, LAVA_MAX)
                px = room.wx + rng.randint(96, max(97, room.w - size - 96))
                py = room.wy + rng.randint(96, max(97, room.d - size - 96))
                candidate = SolidBox(
                    px - 16, py - 16, fz,
                    px + size + 16, py + size + 16, fz + 64,
                )
                if not self._structure_is_clear(candidate):
                    continue
                # Lava and its rim reward form one optional transaction.  The
                # candidate must participate in the final-geometry standing
                # check, but no brush or hazard may be committed until both
                # sides are admissible.  A failed reward search rolls back
                # only the just-appended blocker and tries another candidate.
                self._admit_spawn_blocker(candidate, required=True)
                left = px - 48
                right = px + size + 48
                primary, secondary = (
                    (left, right) if rng.random() < 0.5 else (right, left)
                )
                mega_origin = None
                for preferred_x in (primary, secondary):
                    mega_origin = self._floor_item_spot(
                        room, preferred_x, py + size // 2, 192
                    )
                    if mega_origin is not None:
                        break
                if mega_origin is None:
                    removed = self.spawn_blockers.pop()
                    if removed is not candidate:
                        raise RuntimeError(
                            "lava placement transaction lost blocker ownership"
                        )
                    continue
                placement = (size, px, py, candidate, mega_origin)
                break
            if placement is None:
                continue
            size, px, py, _box, mega_origin = placement
            # Pool sits on the floor with a low rim so it reads as terrain
            w.add_brush(px - 16, py - 16, fz, px + size + 16, py + size + 16, fz + 12,
                        tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['trim'])
            w.add_brush(px, py, fz + 12, px + size, py + size, fz + 12 + LAVA_DEPTH,
                        tf=T_LAVA, tc=T_LAVA, tw=T_LAVA,
                        surf_flags=SURF_LIGHT | SURF_WARP, value=120,
                        contents=CONTENTS_LAVA)
            # The padded rim/clearance box is a placement blocker, not the
            # hazardous volume.  Claims and Dyn priors bind the exact emitted
            # CONTENTS_LAVA brush so compiled-world probes do not inherit 16u
            # of solid rim or 44u of empty air.
            self.lava_pools.append(SolidBox(
                px, py, fz + 12,
                px + size, py + size, fz + 12 + LAVA_DEPTH,
            ))
            # Mega health on the rim: worth the burn risk
            self._reserve_item_origin(mega_origin, "lava-rim mega health")
            mx, my, mz = mega_origin
            w.add_entity("item_health_mega",
                         {"origin": f"{mx} {my} {mz}"})
            self._placed_loot.append(("item_health_mega", mx, my, mz))

    # ----- entity placement -----

    def _item_origin_available(self, origin: Tuple[int, int, int]) -> bool:
        """Return whether an item can exclusively claim this exact origin."""
        return (
            origin not in self._item_origins
            and origin not in self.spawn_points
            and origin not in self._spawn_protected_witnesses
        )

    def _reserve_item_origin(self, origin: Tuple[int, int, int], label: str) -> None:
        """Reserve an emitted item origin, failing closed on any collision."""
        if origin in self._spawn_protected_witnesses:
            raise RuntimeError(
                f"item origin enters protected spawn witness for {label}: {origin}"
            )
        if not self._item_origin_available(origin):
            raise RuntimeError(f"duplicate item origin for {label}: {origin}")
        self._item_origins.add(origin)

    def _inside_room_for_spawn(self, room: Room, x: int, y: int) -> bool:
        return (
            room.wx + SPAWN_EDGE_MARGIN <= x <= room.wx + room.w - SPAWN_EDGE_MARGIN and
            room.wy + SPAWN_EDGE_MARGIN <= y <= room.wy + room.d - SPAWN_EDGE_MARGIN
        )

    def _spawn_overlaps_blocker(self, x: int, y: int, z: int) -> bool:
        half = PLAYER_XY_HALF + SPAWN_SOLID_MARGIN
        px0, px1 = x - half, x + half
        py0, py1 = y - half, y + half
        pz0, pz1 = z + PLAYER_MINS_Z, z + PLAYER_MAXS_Z
        for box in self.spawn_blockers:
            if box.x1 <= px0 or box.x0 >= px1:
                continue
            if box.y1 <= py0 or box.y0 >= py1:
                continue
            if box.z1 <= pz0 or box.z0 >= pz1:
                continue
            return True
        return False

    def _standing_hull_overlaps_blocker(self, x: int, y: int, z: int) -> bool:
        """Match the exact Quake standing hull used by compiled claim probes."""
        hull = SolidBox(
            x - PLAYER_XY_HALF, y - PLAYER_XY_HALF, z + PLAYER_MINS_Z,
            x + PLAYER_XY_HALF, y + PLAYER_XY_HALF, z + PLAYER_MAXS_Z,
        )
        return any(
            self._boxes_overlap(hull, blocker)
            for blocker in self.spawn_blockers
        )

    def _player_column_is_clear(self, x: float, y: float, floor_z: int) -> bool:
        """Check the source interval required by the compiled 96u probe."""
        px0, px1 = x - PLAYER_XY_HALF, x + PLAYER_XY_HALF
        py0, py1 = y - PLAYER_XY_HALF, y + PLAYER_XY_HALF
        column_bottom = floor_z + 1
        column_top = floor_z + SPAWN_COMPILED_COLUMN_HEIGHT
        for box in self.spawn_blockers:
            if box.x1 <= px0 or box.x0 >= px1:
                continue
            if box.y1 <= py0 or box.y0 >= py1:
                continue
            if box.z1 <= column_bottom or box.z0 >= column_top:
                continue
            return False
        return True

    def _spawn_has_escape(self, room: Room, x: int, y: int) -> bool:
        """Require one clear, supported 96u horizontal route from a start."""
        floor_z = room.floor_z
        if not self._player_column_is_clear(x, y, floor_z):
            return False
        directions = (
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (math.sqrt(0.5), math.sqrt(0.5)),
            (math.sqrt(0.5), -math.sqrt(0.5)),
            (-math.sqrt(0.5), math.sqrt(0.5)),
            (-math.sqrt(0.5), -math.sqrt(0.5)),
        )
        for dx, dy in directions:
            clear = True
            for distance in range(
                    SPAWN_ESCAPE_STEP,
                    SPAWN_ESCAPE_DISTANCE + SPAWN_ESCAPE_STEP,
                    SPAWN_ESCAPE_STEP):
                px = x + dx * distance
                py = y + dy * distance
                if not (
                    room.wx + PLAYER_XY_HALF <= px <=
                    room.wx + room.w - PLAYER_XY_HALF and
                    room.wy + PLAYER_XY_HALF <= py <=
                    room.wy + room.d - PLAYER_XY_HALF
                ):
                    clear = False
                    break
                if not self._player_column_is_clear(px, py, floor_z):
                    clear = False
                    break
            if clear:
                return True
        return False

    def _spawn_is_clear(self, room: Room, x: int, y: int, z: int) -> bool:
        if not self._spawn_is_locally_clear(room, x, y, z):
            return False
        for sx, sy, _ in self.spawn_points:
            if math.hypot(x - sx, y - sy) < MIN_SPAWN_SEPARATION:
                return False
        return True

    def _spawn_is_locally_clear(
        self, room: Room, x: int, y: int, z: int,
    ) -> bool:
        """Check every spawn condition except separation from other starts."""
        if not self._inside_room_for_spawn(room, x, y):
            return False
        if self._spawn_overlaps_blocker(x, y, z):
            return False
        if not self._spawn_has_escape(room, x, y):
            return False
        if (x, y, z) in self._item_origins:
            return False
        return True

    def _choose_spawn(self, room: Room) -> Optional[Tuple[int, int, int]]:
        rng = self.rng
        margin = min(160, max(64, min(room.w, room.d) // 4))
        x0 = room.wx + margin
        x1 = room.wx + room.w - margin
        y0 = room.wy + margin
        y1 = room.wy + room.d - margin
        fz = room.floor_z + 24

        best = (room.wx + room.w // 2, room.wy + room.d // 2, fz)
        best_dist = -1.0
        for _ in range(96):
            x = rng.randint(min(x0, x1), max(x0, x1))
            y = rng.randint(min(y0, y1), max(y0, y1))
            if self._spawn_is_clear(room, x, y, fz):
                return x, y, fz
            if not self._inside_room_for_spawn(room, x, y):
                continue
            if self._spawn_overlaps_blocker(x, y, fz):
                continue
            if not self._spawn_has_escape(room, x, y):
                continue
            if self.spawn_points:
                dist = min(math.hypot(x - sx, y - sy) for sx, sy, _ in self.spawn_points)
            else:
                dist = float("inf")
            if dist > best_dist:
                best_dist = dist
                best = (x, y, fz)

        if best_dist >= MIN_SPAWN_SEPARATION:
            return best
        return None

    def _combat_spawn_room(self) -> Room:
        arenas = [room for room in self.rooms if room.kind == 'arena']
        rooms = arenas or self.rooms
        return max(rooms, key=lambda room: room.w * room.d)

    def _combat_spawn_candidate(self, room: Room, base_angle: float, radius: float) -> Optional[Tuple[int, int, int]]:
        cx = room.wx + room.w // 2
        cy = room.wy + room.d // 2
        fz = room.floor_z + 24
        radius_steps = [1.0, 0.9, 0.8, 0.7, 1.05, 0.6]
        angle_offsets = [0, 12, -12, 24, -24, 36, -36, 48, -48, 60, -60, 75, -75]

        for scale in radius_steps:
            r = max(256.0, min(radius * scale, radius))
            for offset in angle_offsets:
                angle = math.radians(base_angle + offset)
                x = int(round(cx + math.cos(angle) * r))
                y = int(round(cy + math.sin(angle) * r))
                if self._spawn_is_clear(room, x, y, fz):
                    return x, y, fz
        return self._choose_spawn(room)

    def _try_combat_spawns_in(self, room: Room) -> Optional[List[Tuple[int, int, int, int]]]:
        """Try to place the full separated spawn set in one room. Returns
        [(x, y, z, yaw)] or None, leaving self.spawn_points untouched on failure."""
        cx = room.wx + room.w // 2
        cy = room.wy + room.d // 2
        max_radius = min(room.w, room.d) / 2 - SPAWN_EDGE_MARGIN
        radius = min(640.0, max_radius)
        if radius < 272.0:
            return None

        placed: List[Tuple[int, int, int, int]] = []
        saved = list(self.spawn_points)
        # Even ring spacing avoids the old four-point cardinal cluster and
        # leaves two spare starts in a six-player match.
        for index in range(DM_SPAWN_COUNT):
            base_angle = 180.0 + index * (360.0 / DM_SPAWN_COUNT)
            candidate = self._combat_spawn_candidate(room, base_angle, radius)
            if candidate is None:
                self.spawn_points = saved
                return None
            x, y, fz = candidate
            yaw = int(round((math.degrees(math.atan2(cy - y, cx - x)) + 360.0) % 360.0))
            self.spawn_points.append((x, y, fz))
            placed.append((x, y, fz, yaw))
        return placed

    def _emit_spawn_entities(self, placed: List[Tuple[int, int, int, int]]):
        for x, y, fz, yaw in placed:
            self.writer.add_entity("info_player_deathmatch", {
                "origin": f"{x} {y} {fz}",
                "angle": str(yaw),
            })

    def _spawn_span_ok(self, placed) -> bool:
        xs = [p[0] for p in placed]
        ys = [p[1] for p in placed]
        return (max(xs) - min(xs)) >= 1024 and (max(ys) - min(ys)) >= 1024

    def _shared_spawn_source_component(
        self, origins: Sequence[Tuple[int, int, int]],
    ) -> Optional[int]:
        """Return the one conservative standing component owning all starts."""
        if len(origins) != DM_SPAWN_COUNT:
            return None
        from maps.routes import source_endpoint_components

        components = source_endpoint_components(
            self.rooms, origins, self.spawn_blockers, self.lava_pools,
        )
        admitted = [components.get(index) for index in range(len(origins))]
        if any(component is None for component in admitted):
            return None
        distinct = set(admitted)
        return admitted[0] if len(distinct) == 1 else None

    def _spawn_candidate_pool(self) -> List[Tuple[int, int, int]]:
        """Enumerate deterministic, locally legal starts across final geometry.

        The 64-unit lattice is finer than the 384-unit separation gate and is
        anchored independently in every room so narrow connected floor bands
        remain represented.  Global separation is applied only after source
        components are known.
        """
        candidates: set[Tuple[int, int, int]] = set()
        candidate_step = 64
        for room in sorted(
            self.rooms,
            key=lambda value: (
                value.floor_z, value.wy, value.wx, value.d, value.w,
            ),
        ):
            x0 = room.wx + SPAWN_EDGE_MARGIN
            x1 = room.wx + room.w - SPAWN_EDGE_MARGIN
            y0 = room.wy + SPAWN_EDGE_MARGIN
            y1 = room.wy + room.d - SPAWN_EDGE_MARGIN
            if x0 > x1 or y0 > y1:
                continue
            z = room.floor_z + 24
            for x in range(x0, x1 + 1, candidate_step):
                for y in range(y0, y1 + 1, candidate_step):
                    if self._spawn_is_locally_clear(room, x, y, z):
                        candidates.add((x, y, z))
        return sorted(candidates, key=lambda point: (point[2], point[1], point[0]))

    def _greedy_component_spawn_set(
        self, candidates: Sequence[Tuple[int, int, int]],
    ) -> Optional[List[Tuple[int, int, int]]]:
        """Find a separated, map-spanning eight-start subset deterministically."""
        if len(candidates) < DM_SPAWN_COUNT:
            return None

        ordered = sorted(candidates, key=lambda point: (point[2], point[1], point[0]))
        extrema = {
            min(ordered, key=lambda point: (point[0], point[1], point[2])),
            max(ordered, key=lambda point: (point[0], -point[1], -point[2])),
            min(ordered, key=lambda point: (point[1], point[0], point[2])),
            max(ordered, key=lambda point: (point[1], -point[0], -point[2])),
            min(ordered, key=lambda point: (point[0] + point[1], point[2])),
            max(ordered, key=lambda point: (point[0] + point[1], -point[2])),
            min(ordered, key=lambda point: (point[0] - point[1], point[2])),
            max(ordered, key=lambda point: (point[0] - point[1], -point[2])),
        }
        # Cover the interior as well as extrema; a blocker beside an extreme
        # must not make the only greedy attempt determine source publication.
        stride = max(1, len(ordered) // 24)
        seeds = sorted(
            extrema.union(ordered[::stride]),
            key=lambda point: (point[2], point[1], point[0]),
        )

        passing: List[List[Tuple[int, int, int]]] = []
        for seed in seeds:
            chosen = [seed]
            while len(chosen) < DM_SPAWN_COUNT:
                eligible = [
                    point for point in ordered
                    if point not in chosen
                    and all(
                        math.hypot(point[0] - other[0], point[1] - other[1])
                        >= MIN_SPAWN_SEPARATION
                        for other in chosen
                    )
                ]
                if not eligible:
                    break

                def priority(point: Tuple[int, int, int]) -> tuple:
                    trial = [*chosen, point]
                    x_span = max(value[0] for value in trial) - min(
                        value[0] for value in trial
                    )
                    y_span = max(value[1] for value in trial) - min(
                        value[1] for value in trial
                    )
                    capped_x = min(x_span, 1024)
                    capped_y = min(y_span, 1024)
                    nearest = min(
                        math.hypot(point[0] - other[0], point[1] - other[1])
                        for other in chosen
                    )
                    return (
                        capped_x + capped_y,
                        capped_x * capped_y,
                        nearest,
                        -point[2], -point[1], -point[0],
                    )

                chosen.append(max(eligible, key=priority))
            if len(chosen) == DM_SPAWN_COUNT and self._spawn_span_ok(chosen):
                passing.append(chosen)

        if not passing:
            return None

        def result_priority(points: Sequence[Tuple[int, int, int]]) -> tuple:
            minimum_separation = min(
                math.hypot(left[0] - right[0], left[1] - right[1])
                for index, left in enumerate(points)
                for right in points[index + 1:]
            )
            x_span = max(point[0] for point in points) - min(point[0] for point in points)
            y_span = max(point[1] for point in points) - min(point[1] for point in points)
            canonical = tuple(sorted(points, key=lambda point: (point[2], point[1], point[0])))
            return minimum_separation, x_span * y_span, x_span + y_span, canonical

        return list(max(passing, key=result_priority))

    def _component_constrained_combat_spawns(
        self,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Select all starts from one source standing component or fail."""
        from maps.routes import source_endpoint_components

        candidates = self._spawn_candidate_pool()
        components = source_endpoint_components(
            self.rooms, candidates, self.spawn_blockers, self.lava_pools,
        )
        groups: dict[int, List[Tuple[int, int, int]]] = {}
        for index, candidate in enumerate(candidates):
            component = components.get(index)
            if component is not None:
                groups.setdefault(component, []).append(candidate)

        selections = []
        for component in sorted(groups):
            selected = self._greedy_component_spawn_set(groups[component])
            if selected is not None:
                selections.append((component, selected))
        if not selections:
            return None

        # Prefer the widest admitted solution; component ID and coordinates
        # make equal layouts byte-deterministic.
        def selection_priority(value) -> tuple:
            component, selected = value
            x_span = max(point[0] for point in selected) - min(point[0] for point in selected)
            y_span = max(point[1] for point in selected) - min(point[1] for point in selected)
            return (
                x_span * y_span, x_span + y_span, -component,
                tuple(sorted(selected, key=lambda point: (point[2], point[1], point[0]))),
            )

        _component, selected = max(selections, key=selection_priority)
        map_cx = sum(room.wx + room.w // 2 for room in self.rooms) // len(self.rooms)
        map_cy = sum(room.wy + room.d // 2 for room in self.rooms) // len(self.rooms)
        return [
            (
                x, y, z,
                int(round((math.degrees(math.atan2(map_cy - y, map_cx - x)) + 360.0) % 360.0)),
            )
            for x, y, z in selected
        ]

    def _spawn_component_bound_diagnostic(self) -> str:
        """Describe the exact source components available to spawn search."""
        from maps.routes import source_endpoint_components

        candidates = self._spawn_candidate_pool()
        components = source_endpoint_components(
            self.rooms, candidates, self.spawn_blockers, self.lava_pools,
        )
        groups: dict[int, List[Tuple[int, int, int]]] = {}
        for index, candidate in enumerate(candidates):
            component = components.get(index)
            if component is not None:
                groups.setdefault(component, []).append(candidate)

        bounds = []
        for component in sorted(groups):
            points = groups[component]
            x_span = max(point[0] for point in points) - min(
                point[0] for point in points
            )
            y_span = max(point[1] for point in points) - min(
                point[1] for point in points
            )
            bounds.append(
                f"{component}:count={len(points)},span={x_span}x{y_span}"
            )
        return (
            f"legal_candidates={len(candidates)}, "
            f"component_bounds=[{'; '.join(bounds)}]"
        )

    def _emit_arena_cover(self):
        """Extra cover geometry in arenas (on top of the baseline _emit_cover):
        peek-height pillars in the fight space so
        bots get cover-fire / cover-dodge instead of dying in the open. The
        per-map learnability data showed open-with-cover maps are trainable
        while exposed (and lava/maze) maps farm the bot — this gives the
        breakable sightlines that make engagements survivable. Pillars also
        flank lava rims so a knock-in has something to break LOS behind."""
        rng = self.rng
        w = self.writer
        COVER_H = 112      # chest/head height — peek over, break LOS, vault around
        COVER_W = 72
        for room in self.rooms:
            if room.kind != 'arena':
                continue
            fz = room.floor_z
            target = rng.randint(*self.arena_cover_range)
            placed = 0
            for _ in range(target * 6):
                if placed >= target:
                    break
                # mid-ring: away from dead-center (where objectives sit) and
                # the walls, so it shapes the open fighting area.
                cx = room.wx + rng.randint(room.w // 4, max(room.w // 4 + 1, 3 * room.w // 4))
                cy = room.wy + rng.randint(room.d // 4, max(room.d // 4 + 1, 3 * room.d // 4))
                box = SolidBox(cx - COVER_W // 2, cy - COVER_W // 2, fz,
                               cx + COVER_W // 2, cy + COVER_W // 2, fz + COVER_H)
                if not self._structure_is_clear(box):
                    continue
                self._admit_spawn_blocker(box, required=True)
                w.add_brush(box.x0, box.y0, box.z0, box.x1, box.y1, box.z1,
                            tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['wall'])
                self.cover_count += 1
                placed += 1

    def _place_combat_spawns(self):
        """Emit the final-geometry certified deathmatch spawn set.

        Generated layouts consume the eight canonical protected-ring witnesses
        only after their clearance, escape, separation, dual span, and exact
        source component are revalidated. Hand-constructed no-arena fixtures
        retain the same strict component-bound selector. Neither path relaxes
        or rescues a failed layout."""
        self._assert_spawn_protected_capacity()
        if self._spawn_protected_witnesses:
            anchor = self._spawn_protected_anchor
            if anchor is None:
                raise RuntimeError("protected spawn selector lacks its anchor")
            center_x = anchor.wx + anchor.w // 2
            center_y = anchor.wy + anchor.d // 2
            placed = [
                (
                    x, y, z,
                    int(round((math.degrees(math.atan2(
                        center_y - y, center_x - x,
                    )) + 360.0) % 360.0)),
                )
                for x, y, z in self._spawn_protected_witnesses
            ]
            self.spawn_points.extend(
                (x, y, z) for x, y, z, _yaw in placed
            )
            if self._shared_spawn_source_component(self.spawn_points) is None:
                raise RuntimeError(
                    "protected spawn selector lost its source component"
                )
            self._emit_spawn_entities(placed)
            return

        arenas = sorted(
            (r for r in self.rooms if r.kind == 'arena'),
            key=lambda r: r.w * r.d, reverse=True,
        )
        for room in arenas:
            placed = self._try_combat_spawns_in(room)
            if (
                placed
                and self._spawn_span_ok(placed)
                and self._shared_spawn_source_component(
                    [(x, y, z) for x, y, z, _yaw in placed]
                ) is not None
            ):
                self._emit_spawn_entities(placed)
                return
            if placed:
                self.spawn_points = self.spawn_points[:-len(placed)]

        placed = self._component_constrained_combat_spawns()
        if placed is None:
            raise RuntimeError(
                f"could not place {DM_SPAWN_COUNT} clear, separated, "
                "map-spanning deathmatch spawns in one source standing component; "
                f"{self._spawn_component_bound_diagnostic()}"
            )
        self.spawn_points.extend((x, y, z) for x, y, z, _yaw in placed)
        if self._shared_spawn_source_component(self.spawn_points) is None:
            raise RuntimeError("selected deathmatch spawns do not share one source component")
        self._emit_spawn_entities(placed)

    def _floor_item_spot(self, room: Room, xc: int, yc: int,
                         spread: int, *,
                         excluded: Optional[set[Tuple[int, int, int]]] = None,
                         ) -> Optional[Tuple[int, int, int]]:
        """Choose a deterministic standing-clear item origin on a room floor."""
        origin_z = room.floor_z + 24
        candidates = []
        for _ in range(32):
            x = xc + self.rng.randint(-spread, spread)
            y = yc + self.rng.randint(-spread, spread)
            candidates.append((x, y))
        x_start = room.wx + SPAWN_EDGE_MARGIN
        x_stop = room.wx + room.w - SPAWN_EDGE_MARGIN
        y_start = room.wy + SPAWN_EDGE_MARGIN
        y_stop = room.wy + room.d - SPAWN_EDGE_MARGIN
        candidates.extend(sorted(
            (
                (x, y)
                for x in range(x_start, x_stop + 1, 32)
                for y in range(y_start, y_stop + 1, 32)
            ),
            key=lambda point: (
                (point[0] - xc) ** 2 + (point[1] - yc) ** 2,
                point[1], point[0],
            ),
        ))
        excluded = excluded or set()
        for x, y in dict.fromkeys(candidates):
            origin = (x, y, origin_z)
            if origin in excluded or not self._item_origin_available(origin):
                continue
            if not self._inside_room_for_spawn(room, x, y):
                continue
            if self._spawn_overlaps_blocker(x, y, origin_z):
                continue
            if self._player_column_is_clear(x, y, room.floor_z):
                return x, y, origin_z
        return None

    def _place_spawn_route_anchors(self) -> None:
        """Seed real offense/survival routes in one spawn component.

        A vertically layered layout can put every deathmatch start on a floor
        component whose rich item economy lives elsewhere. Route archetypes
        may not relabel unrelated loot to hide that topology. Instead, place
        two offense and two survival pickups in an exact conservative source
        component that owns real spawns, before the heat economy is solved
        around them.
        """
        from maps.routes import source_endpoint_components

        candidate_sites = sorted(
            site for site in self._candidate_sites(TIER_CELL[3])
            if self._item_origin_available(site)
            and all(
                math.dist(site, spawn) >= 64.0
                for spawn in self.spawn_points
            )
        )
        endpoints = [*self.spawn_points, *candidate_sites]
        components = source_endpoint_components(
            self.rooms, endpoints, self.spawn_blockers, self.lava_pools
        )
        spawn_groups: dict[int, List[Tuple[int, int, int]]] = {}
        for index, spawn in enumerate(self.spawn_points):
            component = components.get(index)
            if component is not None:
                spawn_groups.setdefault(component, []).append(spawn)
        if not spawn_groups:
            raise RuntimeError("no deathmatch spawn has a source component")
        selected_component = min(
            spawn_groups,
            key=lambda component: (
                -len(spawn_groups[component]),
                min(spawn_groups[component]),
                component,
            ),
        )
        component_spawns = spawn_groups[selected_component]
        candidate_offset = len(self.spawn_points)
        component_sites = [
            site for offset, site in enumerate(candidate_sites)
            if components.get(candidate_offset + offset) == selected_component
        ]
        component_sites.sort(key=lambda site: (
            min(math.dist(site, spawn) for spawn in component_spawns),
            site[2], site[1], site[0],
        ))
        chosen: List[Tuple[int, int, int]] = []
        for site in component_sites:
            if all(math.dist(site, other) >= 64.0 for other in chosen):
                chosen.append(site)
                if len(chosen) == len(SPAWN_ROUTE_ANCHOR_CLASSES):
                    break
        if len(chosen) != len(SPAWN_ROUTE_ANCHOR_CLASSES):
            raise RuntimeError(
                "could not place offense/survival anchors in a spawn component"
            )
        for name, site in zip(SPAWN_ROUTE_ANCHOR_CLASSES, chosen):
            self._reserve_item_origin(site, f"spawn route anchor {name}")
            self._placed_loot.append((name, *site))
            self.writer.add_entity(
                name, {"origin": f"{site[0]} {site[1]} {site[2]}"}
            )

    def _origin_has_final_standing_floor(
        self, origin: Tuple[int, int, int]
    ) -> bool:
        """Return whether an endpoint has a final source standing witness.

        This mirrors the compiled claim's standing hull, not the generator's
        larger 96u comfort/headroom rule used when choosing new spawns.  A
        legal 56u Quake standing pocket is valid evidence here.
        """
        x, y, z = origin
        for room in self.rooms:
            if z != room.floor_z + 24:
                continue
            if not (
                room.wx <= x <= room.wx + room.w
                and room.wy <= y <= room.wy + room.d
            ):
                continue
            if not self._standing_hull_overlaps_blocker(x, y, z):
                return True
        return False

    def _assert_final_route_endpoints_standing_clear(self) -> None:
        """Reject any source endpoint invalidated by final static geometry."""
        item_endpoints = [
            *((item, x, y, z) for item, x, y, z in self._placed_loot),
            *((item, x, y, z) for item, x, y, z in self._heat_placed),
            *((o["item"], o["x"], o["y"], o["z"]) for o in self.objectives),
        ]
        for item, x, y, z in item_endpoints:
            if not self._origin_has_final_standing_floor((x, y, z)):
                raise RuntimeError(
                    "published item lacks a final standing-floor witness: "
                    f"{item} at {(x, y, z)}"
                )
        for origin in self.spawn_points:
            if not self._origin_has_final_standing_floor(origin):
                raise RuntimeError(
                    "published spawn lacks a final standing-floor witness: "
                    f"{origin}"
                )

    def _materialize_floor_loot_sites(self) -> None:
        """Relocate every structure reward to a reachable floor band.

        Floating platforms, tower tops, and roofs remain traversal geometry,
        but their rewards cannot be emitted until the exact graph proves an
        inbound and return path.  The v1 generator therefore places those
        rewards beside/under the structure on a standing-clear room floor.
        """
        materialized: List[Tuple[str, int, int, int]] = []
        seen: set[Tuple[int, int, int]] = set()
        for kind, x, y, z in self._loot_sites:
            containing = [
                (abs(z - (room.floor_z + 24)), index, room)
                for index, room in enumerate(self.rooms)
                if room.wx <= x <= room.wx + room.w
                and room.wy <= y <= room.wy + room.d
            ]
            for _, _, room in sorted(containing):
                site = self._floor_item_spot(
                    room, x, y, 192, excluded=seen
                )
                if site is None or site in seen:
                    continue
                seen.add(site)
                materialized.append((kind, *site))
                break
        self._loot_sites = materialized

    def _place_entities(self):
        self._place_combat_spawns()
        self._place_spawn_route_anchors()

        # Platform loot sites stay structure-positioned (rewards hook use),
        # but WHAT lands on each is heat-chosen in _heat_place_items.
        for room in self.rooms:
            for p in room.platforms:
                px = (p['x0'] + p['x1']) // 2
                py = (p['y0'] + p['y1']) // 2
                # Above the platform TOP (brush spans z .. z+thick); the item
                # bbox reaches 15 below origin, so z+24 was inside the brush.
                pz = p['z'] + p['thick'] + 24
                self._loot_sites.append(("platform", px, py, pz))

        self._materialize_floor_loot_sites()
        self._heat_place_items()

    def _heat_place_items(self):
        """Coarse-to-fine heat-templated placement (cubic reduction steps).

        Seed the field from everything that already exists (spawns,
        objectives, lava, structure loot), then place floor items tier by
        tier on progressively finer candidate grids. Every placement
        deposits its emission, steering everything placed after it."""
        rng = self.rng
        field = self._rebuild_field()
        self._heat = field

        # Global armour quota (judge: 19 armours vs 12 power weapons = free
        # value everywhere). Shared between structure loot and the tier-2 pass.
        armor_cap = max(2, (len(self._loot_sites) + len(self.rooms) // 2) // 4)
        self._place_structure_loot(field, armor_cap)

        candidate_sites = self._candidate_sites

        n_arena = max(1, sum(1 for r in self.rooms if r.kind == 'arena'))
        budget = {
            1: max(2, n_arena // 2 + 1),
            2: max(4, len(self.rooms) // 2),
        }

        def n_armor():
            return sum(1 for c, *_ in self._placed_loot + self._heat_placed
                       if c in ARMOR_CLASSES)

        def n_class(cls):
            return sum(1 for c, *_ in self._placed_loot + self._heat_placed
                       if c == cls)

        def power_sites():
            return [(x, y, z) for c, x, y, z in
                    self._placed_loot + self._heat_placed if c in POWER_WEAPONS]

        def armor_sites():
            return [(x, y, z) for c, x, y, z in
                    self._placed_loot + self._heat_placed if c in ARMOR_CLASSES]

        for tier in (1, 2):
            cands = [
                site for site in candidate_sites(TIER_CELL[tier])
                if self._item_origin_available(site)
            ]
            for _ in range(budget[tier]):
                if not cands:
                    break
                names = [n for n, t in ITEM_TEMPLATES.items()
                         if t["tier"] == tier and not t.get("rune")
                         and not (n in ARMOR_CLASSES and n_armor() >= armor_cap)
                         and not (n.startswith("weapon_")
                                  and n_class(n) >= WEAPON_CLASS_CAP)]
                if not names:
                    break
                name = rng.choice(names)
                t = ITEM_TEMPLATES[name]
                pool = rng.sample(cands, min(48, len(cands)))
                if name in POWER_WEAPONS:
                    # Hard separation between control points, and never on
                    # top of an already-placed counter (the floor is
                    # bidirectional — whichever lands second must respect it)
                    held_p, held_a = power_sites(), armor_sites()
                    spread = [p for p in pool
                              if all(math.dist(p, s) >= POWER_MIN_SEPARATION
                                     for s in held_p)
                              and all(math.dist(p, s) >= ARMOR_MIN_FROM_POWER
                                      for s in held_a)]
                    pool = spread or pool
                elif name in ARMOR_CLASSES:
                    # Counters answer threats from outside melee range
                    held = power_sites()
                    spaced = [p for p in pool if all(
                        math.dist(p, s) >= ARMOR_MIN_FROM_POWER for s in held)]
                    pool = spaced or pool
                best = max(pool, key=lambda p:
                           field.score(t["pull"], *p) + rng.uniform(0.0, 0.15))
                x, y, z = best
                for ch, amt in t["emit"].items():
                    field.deposit(ch, x, y, z, amt, t["radius"])
                self._reserve_item_origin(best, f"heat tier {tier} {name}")
                self._heat_placed.append((name, x, y, z))
                cands.remove(best)

        # Tier 3 is demand-driven: ammo supply follows the weapons actually
        # placed (~1 box per 2 weapons of a class — judge: 6 railguns shared
        # 1 slug box), plus a guaranteed health floor.
        demand: List[str] = []
        weapon_counts: dict = {}
        for c, *_ in self._placed_loot + self._heat_placed:
            if c in AMMO_FOR_WEAPON:
                weapon_counts[c] = weapon_counts.get(c, 0) + 1
        for wcls, n in sorted(weapon_counts.items()):
            # Power weapons sustain near 1:1 (contested spots burn ammo);
            # mid-tier at ~1 box per 2 weapons.
            ratio = -(-3 * n // 4) if wcls in POWER_WEAPONS else (n + 1) // 2
            demand += [AMMO_FOR_WEAPON[wcls]] * max(1, ratio)
        demand += ["item_health"] * max(2, len(self.rooms) // 3)
        demand += ["item_health_large", "item_adrenaline"]
        rng.shuffle(demand)

        cands = [
            site for site in candidate_sites(TIER_CELL[3])
            if self._item_origin_available(site)
        ]
        for name in demand:
            if not cands:
                break
            t = ITEM_TEMPLATES[name]
            pool = rng.sample(cands, min(48, len(cands)))
            best = max(pool, key=lambda p:
                       field.score(t["pull"], *p) + rng.uniform(0.0, 0.15))
            x, y, z = best
            for ch, amt in t["emit"].items():
                field.deposit(ch, x, y, z, amt, t["radius"])
            self._reserve_item_origin(best, f"heat tier 3 {name}")
            self._heat_placed.append((name, x, y, z))
            cands.remove(best)

        # Runes are placed ONLY in the rune gym (controlled teaching ground).
        # Normal maps leave runes to the mod's dynamic random spawner — that
        # randomness is intentional entropy the bot must generalize against;
        # static-placing them would make spawn points memorizable.
        # In the gym we CLUSTER all 5 in one spot (a "rune rack") so switching
        # is one short step — the lesson is the DECISION (which rune fits my
        # win_margin now), not the travel.
        if self.gym:
            rune_cands = candidate_sites(TIER_CELL[2])
            runes = ("rune_strength", "rune_haste", "rune_regen",
                     "rune_vampire", "rune_resist")
            offs = [(-48, 0), (48, 0), (0, -48), (0, 48), (0, 0)]
            rune_cands = [
                center for center in rune_cands
                if all(self._item_origin_available(
                    (center[0] + dx, center[1] + dy, center[2])
                ) for dx, dy in offs)
            ]
            if rune_cands:
                cx, cy, cz = rng.choice(rune_cands)   # one central rack site
                for name, (dx, dy) in zip(runes, offs):
                    x, y, z = cx + dx, cy + dy, cz
                    t = ITEM_TEMPLATES[name]
                    for ch, amt in t["emit"].items():
                        field.deposit(ch, x, y, z, amt, t["radius"])
                    self._reserve_item_origin((x, y, z), f"rune rack {name}")
                    self._heat_placed.append((name, x, y, z))

        # Radiative sanity check: with every source placed, re-evaluate each
        # item against the FULL field (minus its own emission) and relocate
        # anything that landed incoherently — greedy placement can't see heat
        # deposited after it. Iterate until stable: coherence by algorithm.
        self.relax_moves = self._relax_placements()

        for name, x, y, z in self._heat_placed:
            self.writer.add_entity(name, {"origin": f"{x} {y} {z}"})

    def _place_structure_loot(self, field: "HeatField", armor_cap: int):
        """Heat-driven loot selection for structure sites (tower tops,
        platforms). Positions are fixed by geometry; the field chooses the
        item, so the economy reacts: a railgun on one tower repels another
        from the tower next door, armour answers weapon heat from a separate
        position instead of being welded to it at +32, and the armour quota
        keeps counters contested instead of free."""
        rng = self.rng

        def n_armor():
            return sum(1 for c, *_ in self._placed_loot if c in ARMOR_CLASSES)

        def n_class(cls):
            return sum(1 for c, *_ in self._placed_loot if c == cls)

        # Towers first (highest exposure should claim the highest value),
        # then platforms in geometry order so repulsion sees neighbours.
        sites = ([s for s in self._loot_sites if s[0] == "tower"]
                 + [s for s in self._loot_sites if s[0] != "tower"])
        for kind, x, y, z in sites:
            pool = TOWER_LOOT_POOL if kind == "tower" else PLATFORM_LOOT_POOL
            names = [n for n in pool
                     if not (n in ARMOR_CLASSES and n_armor() >= armor_cap)
                     and not (n.startswith("weapon_")
                              and n_class(n) >= WEAPON_CLASS_CAP)]
            if not names:
                names = ["item_health_large"]
            held_power = [(px, py, pz) for c, px, py, pz in self._placed_loot
                          if c in POWER_WEAPONS]
            held_armor = [(px, py, pz) for c, px, py, pz in self._placed_loot
                          if c in ARMOR_CLASSES]

            def score(n):
                s = field.score(ITEM_TEMPLATES[n]["pull"], x, y, z)
                if n in POWER_WEAPONS and (
                        any(math.dist((x, y, z), h) < POWER_MIN_SEPARATION
                            for h in held_power)
                        or any(math.dist((x, y, z), h) < ARMOR_MIN_FROM_POWER
                               for h in held_armor)):
                    s -= 2.0
                if n in ARMOR_CLASSES and any(
                        math.dist((x, y, z), h) < ARMOR_MIN_FROM_POWER
                        for h in held_power):
                    s -= 2.0   # counters answer threats, never sit on them
                return s + rng.uniform(0.0, 0.1)

            name = max(names, key=score)
            t = ITEM_TEMPLATES[name]
            for ch, amt in t["emit"].items():
                field.deposit(ch, x, y, z, amt, t["radius"])
            self._reserve_item_origin((x, y, z), f"structure loot {name}")
            self.writer.add_entity(name, {"origin": f"{x} {y} {z}"})
            self._placed_loot.append((name, x, y, z))

    def _rebuild_field(self, skip_idx: Optional[int] = None) -> "HeatField":
        field = HeatField(cell=128)
        for x, y, fz in self.spawn_points:
            field.deposit("spawn", x, y, fz, 1.0, 600)
        for o in self.objectives:
            field.deposit("objective", o["x"], o["y"], o["z"], 1.0, 800)
        for b in self.lava_pools:
            field.deposit("danger", (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2,
                          b.z0, 1.0, 500)
        for src in self._observed_heat:
            field.deposit(src["channel"], src["x"], src["y"], src["z"],
                          src["amount"], src["radius"])
        for cls, x, y, z in self._placed_loot:
            t = ITEM_TEMPLATES.get(cls)
            if t:
                for ch, amt in t["emit"].items():
                    field.deposit(ch, x, y, z, amt, t["radius"])
        for i, (cls, x, y, z) in enumerate(self._heat_placed):
            if i == skip_idx:
                continue
            t = ITEM_TEMPLATES.get(cls)
            if t:
                for ch, amt in t["emit"].items():
                    field.deposit(ch, x, y, z, amt, t["radius"])
        return field

    def _relax_placements(self, sweeps: int = 3, eps: float = 0.12) -> int:
        rng = self.rng
        moves = 0
        sites_cache = {}
        for _ in range(sweeps):
            moved_this_sweep = 0
            for idx in range(len(self._heat_placed)):
                name, x, y, z = self._heat_placed[idx]
                if name.startswith("rune_"):
                    continue   # gym rune rack stays clustered — don't relax apart
                t = ITEM_TEMPLATES[name]
                field = self._rebuild_field(skip_idx=idx)
                cur = field.score(t["pull"], x, y, z)
                stride = TIER_CELL[t["tier"]]
                if stride not in sites_cache:
                    sites_cache[stride] = self._candidate_sites(stride)
                available = [
                    site for site in sites_cache[stride]
                    if self._item_origin_available(site)
                ]
                if not available:
                    continue
                pool = rng.sample(available, min(48, len(available)))
                # Relocation honours the same hard floors as placement —
                # without them the relax pass walked armour right back
                # onto the weapon heat it is meant to answer from afar.
                others = [*self._placed_loot, *(
                    placement for other_idx, placement
                    in enumerate(self._heat_placed) if other_idx != idx
                )]
                held_p = [(px, py, pz) for c, px, py, pz in others
                          if c in POWER_WEAPONS]
                held_a = [(px, py, pz) for c, px, py, pz in others
                          if c in ARMOR_CLASSES]
                if name in ARMOR_CLASSES:
                    spaced = [p for p in pool if all(
                        math.dist(p, s) >= ARMOR_MIN_FROM_POWER for s in held_p)]
                    pool = spaced or pool
                elif name in POWER_WEAPONS:
                    spaced = [p for p in pool
                              if all(math.dist(p, s) >= POWER_MIN_SEPARATION
                                     for s in held_p)
                              and all(math.dist(p, s) >= ARMOR_MIN_FROM_POWER
                                      for s in held_a)]
                    pool = spaced or pool
                best = max(pool, key=lambda p: field.score(t["pull"], *p))
                if field.score(t["pull"], *best) > cur + eps:
                    previous = (x, y, z)
                    self._item_origins.remove(previous)
                    self._reserve_item_origin(best, f"relaxed {name}")
                    self._heat_placed[idx] = (name, best[0], best[1], best[2])
                    moved_this_sweep += 1
            moves += moved_this_sweep
            if moved_this_sweep == 0:
                break
        return moves

    def _candidate_sites(self, stride: int):
        out = []
        for room in self.rooms:
            fz = room.floor_z + 24
            x = room.wx + stride // 2
            while x < room.wx + room.w - 32:
                y = room.wy + stride // 2
                while y < room.wy + room.d - 32:
                    if not self._spawn_overlaps_blocker(x, y, fz):
                        out.append((x, y, fz))
                    y += stride
                x += stride
        # Overlapping generated rooms may contribute the same physical floor
        # point.  Preserve first-seen order while exposing it only once.
        return list(dict.fromkeys(out))

    # ----- measurable base-floor lighting -----

    def _build_floor_light_regions(self) -> List[Tuple[FloorLightRegion, int]]:
        """Partition spawn-clear base floors into deterministic light regions.

        Regions align to the generator's 512-unit world grid.  Samples retain
        the normal spawn bbox clearance, so cover, lava, walls, towers, and
        buried terrace overlaps do not create artificial lighting obligations.
        Duplicate room overlaps collapse to one physical floor region.
        """
        pending: dict = {}
        for room in self.rooms:
            for x0 in range(room.wx, room.wx + room.w, LIGHT_REGION_SIZE):
                x1 = min(x0 + LIGHT_REGION_SIZE, room.wx + room.w)
                for y0 in range(room.wy, room.wy + room.d, LIGHT_REGION_SIZE):
                    y1 = min(y0 + LIGHT_REGION_SIZE, room.wy + room.d)
                    key = (x0, y0, x1, y1, room.floor_z)
                    entry = pending.setdefault(key, {
                        "ceiling_z": room.ceil_z,
                        "samples": set(),
                    })
                    entry["ceiling_z"] = min(entry["ceiling_z"], room.ceil_z)
                    x = x0 + LIGHT_SAMPLE_SPACING // 2
                    while x < x1:
                        y = y0 + LIGHT_SAMPLE_SPACING // 2
                        while y < y1:
                            origin_z = room.floor_z + 24
                            if (self._inside_room_for_spawn(room, x, y) and
                                    not self._spawn_overlaps_blocker(x, y, origin_z)):
                                entry["samples"].add(
                                    (float(x), float(y), float(room.floor_z + 1))
                                )
                            y += LIGHT_SAMPLE_SPACING
                        x += LIGHT_SAMPLE_SPACING

        # Randomly selected DM origins need the same guarantee even when a
        # narrow platform shadow happens to fall between regular grid probes.
        for spawn_x, spawn_y, spawn_origin_z in self.spawn_points:
            floor_z = spawn_origin_z - 24
            for (x0, y0, x1, y1, region_floor_z), entry in pending.items():
                if (region_floor_z == floor_z and
                        x0 <= spawn_x < x1 and y0 <= spawn_y < y1):
                    entry["samples"].add(
                        (float(spawn_x), float(spawn_y), float(floor_z + 1))
                    )

        regions: List[Tuple[FloorLightRegion, int]] = []
        for (x0, y0, x1, y1, floor_z), entry in sorted(pending.items()):
            samples = tuple(sorted(entry["samples"]))
            if not samples:
                continue
            region = FloorLightRegion(
                region_id=f"floor_{x0}_{y0}_{floor_z}",
                bounds=(x0, y0, x1, y1),
                floor_z=floor_z,
                samples=samples,
            )
            regions.append((region, int(entry["ceiling_z"])))
        return regions

    def _add_floor_light(self, region: FloorLightRegion, kind: str,
                         origin: Tuple[float, float, float], value: int):
        source = LightSource(
            region_id=region.region_id,
            kind=kind,
            origin=origin,
            radius=float(FLOOR_LIGHT_RADIUS),
            value=value,
        )
        self.light_sources.append(source)
        ox, oy, oz = source.origin
        self.writer.add_entity("light", {
            "origin": f"{ox:g} {oy:g} {oz:g}",
            "light": str(source.value),
            "_color": "1.0 0.94 0.82",
            "_ml_floor_light": "1",
            "_ml_region": source.region_id,
            "_ml_kind": source.kind,
            "_ml_radius": f"{source.radius:g}",
        })

    def _ensure_spawn_eye_lighting(self) -> None:
        """Give every authored spawn an exact source-side visible light.

        The ordinary floor contract samples the walking surface. Promotion
        instead traces from the linked player eye, so this is a separate and
        stricter obligation. A missing witness receives one deterministic
        qualified light in the already-admitted spawn column.
        """
        self.spawn_eye_samples = []
        regions_by_floor = sorted(
            self.light_regions,
            key=lambda region: (
                region.floor_z, region.bounds[1], region.bounds[0],
                region.region_id,
            ),
        )
        for spawn_index, (spawn_x, spawn_y, spawn_z) in enumerate(
            self.spawn_points
        ):
            eye = (
                float(spawn_x), float(spawn_y),
                float(spawn_z + SPAWN_LIGHT_EYE_OFFSET),
            )
            self.spawn_eye_samples.append(eye)
            visible = any(
                source.value >= MIN_FLOOR_LIGHT_VALUE
                and light_reaches_spawn_eye(
                    source, eye, self.writer.solid_boxes,
                )
                for source in self.light_sources
            )
            if visible:
                continue
            floor_z = spawn_z + PLAYER_MINS_Z
            region = next(
                (
                    candidate for candidate in regions_by_floor
                    if candidate.floor_z == floor_z
                    and candidate.bounds[0] <= spawn_x < candidate.bounds[2]
                    and candidate.bounds[1] <= spawn_y < candidate.bounds[3]
                ),
                None,
            )
            if region is None:
                raise RuntimeError(
                    f"spawn {spawn_index} has no floor-light region"
                )
            self._add_floor_light(
                region, "spawn_column",
                (
                    float(spawn_x), float(spawn_y),
                    float(spawn_z + SPAWN_COLUMN_LIGHT_OFFSET),
                ),
                UNDER_PLATFORM_LIGHT_VALUE,
            )
            if not light_reaches_spawn_eye(
                self.light_sources[-1], eye, self.writer.solid_boxes,
            ):
                raise RuntimeError(
                    f"spawn {spawn_index} has no unobstructed source light"
                )

    @staticmethod
    def _sample_under_box(sample: Tuple[float, float, float], box: SolidBox) -> bool:
        x, y, z = sample
        return box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1 and z < box.z0

    def _emit_floor_lighting(self):
        """Guarantee direct overhead light across every spawn-clear region.

        One ceiling source covers each open region.  Where a platform or
        building roof shadows base floor, a source is added below that
        occluder.  A deterministic final fill pass handles overlapping roofs
        and edge cases until the measurable coverage floor is met.
        """
        planned = self._build_floor_light_regions()
        self.light_regions = [region for region, _ceiling_z in planned]

        for region, ceiling_z in planned:
            region_occluders = [
                box for box in self.light_occluders
                if box.x1 > region.bounds[0] and box.x0 < region.bounds[2]
                and box.y1 > region.bounds[1] and box.y0 < region.bounds[3]
                and box.z0 > region.floor_z + PLAYER_MAXS_Z
            ]
            # Pick an already spawn-clear XY point nearest the tile centre so
            # the point entity cannot land inside a tower or wall.
            cx = (region.bounds[0] + region.bounds[2]) / 2.0
            cy = (region.bounds[1] + region.bounds[3]) / 2.0
            overhead_xy = min(
                region.samples,
                key=lambda sample: ((sample[0] - cx) ** 2 +
                                    (sample[1] - cy) ** 2, sample),
            )
            overhead_z = max(
                float(region.floor_z + PLAYER_H + 24),
                float(ceiling_z - 24),
            )
            self._add_floor_light(
                region, "overhead",
                (overhead_xy[0], overhead_xy[1], overhead_z),
                OVERHEAD_LIGHT_VALUE,
            )

            for box in region_occluders:
                shadowed = [
                    sample for sample in region.samples
                    if self._sample_under_box(sample, box)
                ]
                if not shadowed:
                    continue
                bx = (box.x0 + box.x1) / 2.0
                by = (box.y0 + box.y1) / 2.0
                under_xy = min(
                    shadowed,
                    key=lambda sample: ((sample[0] - bx) ** 2 +
                                        (sample[1] - by) ** 2, sample),
                )
                under_z = max(
                    float(region.floor_z + PLAYER_H + 16),
                    float(box.z0 - 24),
                )
                # Keep the fill source strictly below the occluding face.
                under_z = min(under_z, float(box.z0 - 8))
                self._add_floor_light(
                    region, "under_platform",
                    (under_xy[0], under_xy[1], under_z),
                    UNDER_PLATFORM_LIGHT_VALUE,
                )

            region_sources = [
                source for source in self.light_sources
                if source.region_id == region.region_id
            ]
            coverage = floor_light_coverage(
                region, region_sources, region_occluders
            )
            # A direct vertical fill at an unlit sample is guaranteed to sit
            # below its lowest overhead occluder.  Iterate in sample order to
            # preserve byte-identical output for a fixed seed.
            while coverage < MIN_FLOOR_LIGHT_COVERAGE:
                uncovered = [
                    sample for sample in region.samples
                    if not any(light_reaches_sample(source, sample,
                                                    region_occluders)
                               for source in region_sources)
                ]
                if not uncovered:
                    break
                sample = uncovered[0]
                overhead_faces = [
                    box.z0 for box in region_occluders
                    if self._sample_under_box(sample, box)
                ]
                limit_z = min(overhead_faces) if overhead_faces else ceiling_z
                fill_z = min(
                    float(limit_z - 8),
                    max(float(region.floor_z + PLAYER_H + 16),
                        float(limit_z - 24)),
                )
                kind = "under_platform" if overhead_faces else "overhead_fill"
                self._add_floor_light(
                    region, kind, (sample[0], sample[1], fill_z),
                    (UNDER_PLATFORM_LIGHT_VALUE if overhead_faces
                     else OVERHEAD_LIGHT_VALUE),
                )
                region_sources = [
                    source for source in self.light_sources
                    if source.region_id == region.region_id
                ]
                coverage = floor_light_coverage(
                    region, region_sources, region_occluders
                )
            self.light_coverages[region.region_id] = coverage

        self._ensure_spawn_eye_lighting()

        self.writer.world_props.update({
            "_ml_lighting_version": "2",
            "_ml_light_regions": str(len(self.light_regions)),
            "_ml_floor_lights": str(len(self.light_sources)),
            "_ml_min_light_coverage": f"{MIN_FLOOR_LIGHT_COVERAGE:.3f}",
            "_ml_min_floor_light_value": str(MIN_FLOOR_LIGHT_VALUE),
        })

    def _collect_interior_light_zones(self) -> List[InteriorLightZone]:
        """Build deterministic, enterability-proven room-like light zones."""
        specs = []
        for index, room in enumerate(self.rooms):
            specs.append((
                f"room_{index}",
                "hallway" if room.kind == "corridor" else room.kind,
                (room.wx, room.wy, room.wx + room.w, room.wy + room.d),
                room.floor_z,
                room.ceil_z,
            ))
        specs.extend(self._interior_zone_specs)
        for room_index, room in enumerate(self.rooms):
            for platform_index, platform in enumerate(room.platforms):
                underside_z = platform["z"] - 8
                if underside_z - room.floor_z < MIN_SAFE_HEADROOM:
                    continue
                specs.append((
                    f"under_platform_{room_index}_{platform_index}",
                    "under_platform",
                    (platform["x0"], platform["y0"],
                     platform["x1"], platform["y1"]),
                    room.floor_z,
                    underside_z,
                ))

        zones = []
        for zone_id, kind, bounds, floor_z, ceiling_z in specs:
            x0, y0, x1, y1 = bounds
            if (x1 - x0 < PLAYER_DIAMETER or
                    y1 - y0 < PLAYER_DIAMETER or
                    ceiling_z - floor_z < MIN_SAFE_HEADROOM):
                continue
            samples = sorted({
                sample
                for region in self.light_regions
                if region.floor_z == floor_z
                for sample in region.samples
                if x0 + PLAYER_XY_HALF <= sample[0] <= x1 - PLAYER_XY_HALF
                and y0 + PLAYER_XY_HALF <= sample[1] <= y1 - PLAYER_XY_HALF
                # Floor-light probes guarantee the 56u standing hull only.
                # Interior zones promise the stronger 96u safe-headroom
                # contract, so an overlapping platform may invalidate a
                # coarse sample even though the player does not start solid.
                and self._player_column_is_clear(
                    sample[0], sample[1], floor_z
                )
            })
            if not samples:
                probes = [
                    (float(x), float(y), float(floor_z + 1))
                    for x in range(x0 + PLAYER_XY_HALF,
                                   x1 - PLAYER_XY_HALF + 1,
                                   PLAYER_XY_HALF)
                    for y in range(y0 + PLAYER_XY_HALF,
                                   y1 - PLAYER_XY_HALF + 1,
                                   PLAYER_XY_HALF)
                    if self._player_column_is_clear(x, y, floor_z)
                ]
                if not probes:
                    continue
                samples = probes

            # General room lights belong in open room volume, not accidentally
            # above a building roof/platform. Dedicated building/platform zones
            # intentionally select an anchor beneath their own occluder.
            candidates = samples
            if kind in ("arena", "room", "hallway", "corner_pocket"):
                open_samples = [
                    sample for sample in samples
                    if not any(self._sample_under_box(sample, box)
                               for box in self.light_occluders)
                ]
                if open_samples:
                    candidates = open_samples
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            anchor = min(
                candidates,
                key=lambda sample: (
                    (sample[0] - cx) ** 2 + (sample[1] - cy) ** 2,
                    sample,
                ),
            )
            zones.append(InteriorLightZone(
                zone_id=zone_id,
                kind=kind,
                bounds=bounds,
                floor_z=floor_z,
                ceiling_z=ceiling_z,
                anchor=anchor,
            ))
        return zones

    def _emit_interior_lighting(self):
        """Give every enterable room-like location its own internal source."""
        self.interior_light_zones = self._collect_interior_light_zones()
        promised = {spec[0] for spec in self._interior_zone_specs}
        retained = {zone.zone_id for zone in self.interior_light_zones}
        missing = sorted(promised - retained)
        if missing:
            raise RuntimeError(
                "generated promised interior has no safe light zone: "
                + ", ".join(missing)
            )
        for zone in self.interior_light_zones:
            ax, ay, _az = zone.anchor
            light_z = min(
                float(zone.ceiling_z - 24),
                float(zone.floor_z + MIN_SAFE_HEADROOM),
            )
            source = LightSource(
                region_id=zone.zone_id,
                kind=zone.kind,
                origin=(ax, ay, light_z),
                radius=float(INTERIOR_LIGHT_RADIUS),
                value=INTERIOR_LIGHT_VALUE,
            )
            # This is also a generator invariant: a tagged interior source may
            # never depend on light above the location's roof/platform.
            if not light_reaches_sample(source, zone.anchor,
                                        self.light_occluders):
                raise RuntimeError(
                    f"interior light for {zone.zone_id} is occluded"
                )
            self.interior_light_sources.append(source)
            ox, oy, oz = source.origin
            self.writer.add_entity("light", {
                "origin": f"{ox:g} {oy:g} {oz:g}",
                "light": str(source.value),
                "_color": "1.0 0.96 0.88",
                "_ml_interior_light": "1",
                "_ml_zone": zone.zone_id,
                "_ml_kind": zone.kind,
                "_ml_radius": f"{source.radius:g}",
            })
        self.writer.world_props.update({
            "_ml_interior_zones": str(len(self.interior_light_zones)),
            "_ml_interior_lights": str(len(self.interior_light_sources)),
            "_ml_min_interior_light_value": str(MIN_INTERIOR_LIGHT_VALUE),
            "_ml_min_interior_light_radius": str(MIN_INTERIOR_LIGHT_RADIUS),
        })

    # ----- hook zone annotation -----

    @staticmethod
    def _hook_milliunits(point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return tuple(round(axis * 1000.0) for axis in point)

    @staticmethod
    def _hook_l1_index(point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        # The Atlas origin is snapped to a multiple of 256, so equality in this
        # origin-free 16u index is identical to equality in the final L1 index.
        return tuple(math.floor(axis / 16.0) for axis in point)

    @classmethod
    def _hook_distance_milliunits(
        cls,
        source: Tuple[float, float, float],
        anchor: Tuple[float, float, float],
    ) -> int:
        source_milliunits = cls._hook_milliunits(source)
        anchor_milliunits = cls._hook_milliunits(anchor)
        eye_milliunits = (
            source_milliunits[0],
            source_milliunits[1],
            source_milliunits[2] + HOOK_EYE_Z * 1000,
        )
        return round(math.sqrt(sum(
            (anchor_milliunits[axis] - eye_milliunits[axis]) ** 2
            for axis in range(3)
        )))

    def _authored_floor_origins(self) -> List[Tuple[float, float, float]]:
        """Return deterministic clear-at-source player origins for proposals.

        These are generator facts only. The compiled preflight must prove the
        corresponding standing support, reachability, trace, and trajectory.
        """

        origins = {
            (
                float(sample[0]),
                float(sample[1]),
                float(region.floor_z - PLAYER_MINS_Z) + PMOVE_FIXED_QUANTUM,
            )
            for region in self.light_regions
            for sample in region.samples
        }
        origins.update(
            (float(x), float(y), float(z) + PMOVE_FIXED_QUANTUM)
            for x, y, z in self.spawn_points
        )
        return sorted(origins, key=lambda point: (point[2], point[1], point[0]))

    def _real_ceiling_anchor(
        self, landing: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, float]]:
        """Return the first real ceiling face above a standing floor origin.

        A platform or roof below the room ceiling blocks the proposal instead
        of being relabeled as a ceiling. This prevents both the former phantom
        `cz-WALL_T` anchor and underside-to-top contradictions.
        """

        x, y, z = landing
        overhead = [
            surface for surface in self.horizontal_surfaces
            if surface.box.x0 < x < surface.box.x1
            and surface.box.y0 < y < surface.box.y1
            and surface.box.z0 > z + PLAYER_MAXS_Z
        ]
        if not overhead:
            return None
        first = min(
            overhead,
            key=lambda surface: (
                surface.box.z0,
                surface.box.z1,
                surface.kind,
                surface.surface_id,
            ),
        )
        if first.kind not in {"ceiling", "light_panel"}:
            return None
        return (float(x), float(y), float(first.box.z0))

    @staticmethod
    def _hook_geometry_key(
        geometry: Tuple[
            Tuple[float, float, float], Tuple[float, float, float], int
        ],
    ) -> tuple:
        anchor, landing, flags = geometry
        return (
            landing[2], landing[1], landing[0],
            anchor[2], anchor[1], anchor[0], flags,
        )

    def _diverse_hook_geometry_order(
        self,
        geometries: Sequence[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], int]
        ],
    ) -> List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float], int]
    ]:
        """Order desired L1 cells by spawn proximity, then farthest coverage.

        This is proposal ordering only. Authored room connectivity is never
        treated as movement evidence; compiled CM/Pmove replay remains the
        sole authority. Starting once near each spawn avoids the old
        low-Z/low-Y prefix, while deterministic farthest-point ordering spreads
        the bounded pool over the remaining authored landing cells and floors.
        """

        remaining = sorted(set(geometries), key=self._hook_geometry_key)
        ordered: List[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], int]
        ] = []
        for spawn in sorted(
            self.spawn_points, key=lambda value: (value[2], value[1], value[0])
        ):
            if not remaining:
                break
            sx, sy, sz = spawn
            geometry = min(
                remaining,
                key=lambda value: (
                    value[1][2] != float(sz),
                    abs(value[1][2] - float(sz)),
                    (value[1][0] - sx) ** 2 + (value[1][1] - sy) ** 2,
                    self._hook_geometry_key(value),
                ),
            )
            remaining.remove(geometry)
            ordered.append(geometry)

        if remaining and not ordered:
            ordered.append(remaining.pop(0))
        while remaining:
            selected_cells = [
                self._hook_l1_index(value[1]) for value in ordered
            ]

            def coverage_distance(value) -> int:
                cell = self._hook_l1_index(value[1])
                return min(
                    (cell[0] - prior[0]) ** 2
                    + (cell[1] - prior[1]) ** 2
                    + 4 * (cell[2] - prior[2]) ** 2
                    for prior in selected_cells
                )

            geometry = min(
                remaining,
                key=lambda value: (
                    -coverage_distance(value), self._hook_geometry_key(value)
                ),
            )
            remaining.remove(geometry)
            ordered.append(geometry)
        return ordered

    def _hook_source_priority(
        self,
        source: Tuple[float, float, float],
        landing: Tuple[float, float, float],
        distance_milliunits: int,
    ) -> tuple:
        """Rank likely reachable authored sources without claiming reachability."""

        fixed_spawns = {
            (float(x), float(y), float(z) + PMOVE_FIXED_QUANTUM)
            for x, y, z in self.spawn_points
        }
        spawn_floor_origins = {
            float(z) + PMOVE_FIXED_QUANTUM for _x, _y, z in self.spawn_points
        }
        exact_spawn = source in fixed_spawns
        on_spawn_floor = source[2] in spawn_floor_origins
        on_landing_floor = source[2] == landing[2] + PMOVE_FIXED_QUANTUM
        if exact_spawn:
            authored_tier = 0
        elif on_spawn_floor and on_landing_floor:
            authored_tier = 1
        elif on_spawn_floor:
            authored_tier = 2
        elif on_landing_floor:
            authored_tier = 3
        else:
            authored_tier = 4
        return (
            authored_tier, distance_milliunits,
            source[2], source[1], source[0],
        )

    def _select_hook_sources(
        self,
        sources: Sequence[Tuple[int, Tuple[float, float, float]]],
        landing: Tuple[float, float, float],
    ) -> List[Tuple[int, Tuple[float, float, float]]]:
        """Select both demonstrated local ranks, preferring spawn floors."""

        ranked = sorted(
            sources,
            key=lambda value: self._hook_source_priority(
                value[1], landing, value[0]
            ),
        )
        return ranked[:MAX_HOOK_SOURCES_PER_GEOMETRY]

    @staticmethod
    def _hook_release_window(
        source_rank: int, geometry_ordinal: int
    ) -> Tuple[int, ...]:
        """Cover the useful release interior while retaining both tail bins."""

        if source_rank == 0:
            # Most rank-0 wins are ticks 2..5. One deterministic eighth keeps
            # tick 1 represented without allowing any replay result to steer a
            # particular map row.
            if geometry_ordinal % 8 == 0:
                return HOOK_RELEASE_TICKS[:HOOK_RELEASES_PER_SOURCE]
            return HOOK_RELEASE_TICKS[1:1 + HOOK_RELEASES_PER_SOURCE]
        return HOOK_RELEASE_TICKS[-HOOK_RELEASES_PER_SOURCE:]

    def _annotate_hook_zones(self):
        """Publish source-bound V4 proposals, never traversal authority.

        The pool is deliberately overcomplete. A compiled-world preflight must
        replay one exact source/release schedule, materialize the first six
        proven geometries, and reject the map when six cannot be proved.
        """

        self.hook_zones = []
        self.hook_claim_candidates_v4 = []
        source_origins = self._authored_floor_origins()
        grouped: dict[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], int],
            List[Tuple[int, Tuple[float, float, float]]],
        ] = {}

        for region in sorted(
            self.light_regions,
            key=lambda value: (value.floor_z, value.bounds, value.region_id),
        ):
            for sample in sorted(region.samples):
                landing = (
                    float(sample[0]),
                    float(sample[1]),
                    float(region.floor_z - PLAYER_MINS_Z),
                )
                anchor = self._real_ceiling_anchor(landing)
                if anchor is None:
                    continue
                sources: List[Tuple[int, Tuple[float, float, float]]] = []
                for source in source_origins:
                    if self._hook_l1_index(source) == self._hook_l1_index(landing):
                        continue
                    distance_milliunits = self._hook_distance_milliunits(source, anchor)
                    if not HOOK_MIN * 1000 <= distance_milliunits <= HOOK_MAX * 1000:
                        continue
                    sources.append((distance_milliunits, source))
                if sources:
                    grouped[(anchor, landing, HOOK_CEILING)] = sources

        coordinate_order = sorted(grouped, key=lambda value: (
            value[0][2], value[0][1], value[0][0],
            value[1][2], value[1][1], value[1][0], value[2],
        ))
        diverse_order = self._diverse_hook_geometry_order(list(grouped))
        # Preserve the complete old coordinate-prefix geometry class (winning
        # attestations span ordinals 0..42), then add map-wide cells. Reorder
        # that union by the spawn-seeded farthest traversal so the campaign
        # sees representative floors/XY regions first without map-specific
        # replay rows steering the selection.
        geometry_limit = min(
            len(diverse_order),
            MAX_HOOK_CANDIDATES_V4
            // (MAX_HOOK_SOURCES_PER_GEOMETRY * HOOK_RELEASES_PER_SOURCE),
            MAX_RUNTIME_HOOK_ZONES,
        )
        selected_geometry_set = set(
            coordinate_order[:min(HOOK_COORDINATE_GEOMETRY_FLOOR, geometry_limit)]
        )
        for geometry in diverse_order:
            if len(selected_geometry_set) >= geometry_limit:
                break
            selected_geometry_set.add(geometry)
        geometry_order = [
            geometry for geometry in diverse_order
            if geometry in selected_geometry_set
        ]
        selected_geometry_sources = [
            (geometry, self._select_hook_sources(grouped[geometry], geometry[1]))
            for geometry in geometry_order
        ]
        extra_release_budget = max(
            0,
            MAX_HOOK_CANDIDATES_V4 - sum(
                len(sources) * HOOK_RELEASES_PER_SOURCE
                for _geometry, sources in selected_geometry_sources
            ),
        )
        for geometry_ordinal, (geometry, selected_sources) in enumerate(
            selected_geometry_sources
        ):
            if len(self.hook_zones) >= MAX_RUNTIME_HOOK_ZONES:
                break
            anchor, landing, flags = geometry
            if not selected_sources:
                continue
            candidates = []
            for source_rank, (distance_milliunits, source) in enumerate(
                selected_sources
            ):
                release_window = list(self._hook_release_window(
                    source_rank, geometry_ordinal
                ))
                for release_after_ticks in HOOK_RELEASE_TICKS:
                    if extra_release_budget <= 0:
                        break
                    if release_after_ticks not in release_window:
                        release_window.append(release_after_ticks)
                        extra_release_budget -= 1
                candidates.extend(
                    HookClaimCandidateV4(
                        source=source, trace_target=anchor, landing=landing,
                        release_after_ticks=release_after_ticks,
                        distance_milliunits=distance_milliunits, flags=flags,
                    )
                    for release_after_ticks in release_window
                )
            remaining = MAX_HOOK_CANDIDATES_V4 - len(self.hook_claim_candidates_v4)
            if remaining <= 0:
                break
            candidates = candidates[:remaining]
            if not candidates:
                continue
            for candidate_ordinal, candidate in enumerate(candidates):
                claim_id = (
                    f"hook:{geometry_ordinal:04d}:candidate:{candidate_ordinal:04d}"
                )
                self.hook_claim_candidates_v4.append((claim_id, candidate))
            projection = candidates[0]
            self.hook_zones.append(HookZone(
                anchor=projection.trace_target,
                landing=projection.landing,
                distance=projection.distance_milliunits / 1000.0,
                flags=projection.flags,
            ))

    # ----- skybox (prevents leaks) -----

    def _emit_skybox(self):
        """Enclose the entire map in sky brushes to prevent BSP leaks."""
        w = self.writer
        if not self.rooms:
            return
        all_x = [r.wx for r in self.rooms] + [r.wx + r.w for r in self.rooms]
        all_y = [r.wy for r in self.rooms] + [r.wy + r.d for r in self.rooms]
        all_z = [r.floor_z for r in self.rooms] + [r.ceil_z for r in self.rooms]
        margin = 256
        bx0 = min(all_x) - margin;  bx1 = max(all_x) + margin
        by0 = min(all_y) - margin;  by1 = max(all_y) + margin
        bz0 = min(all_z) - margin;  bz1 = max(all_z) + margin
        T = WALL_T * 4

        w.add_brush(bx0-T, by0-T, bz0-T, bx1+T, by1+T, bz0,
                    tf=T_SKY_SURFACE, tc=T_SKY_SURFACE, tw=T_SKY_SURFACE,
                    surf_flags=SURF_SKY)
        w.add_brush(bx0-T, by0-T, bz1, bx1+T, by1+T, bz1+T,
                    tf=T_SKY_SURFACE, tc=T_SKY_SURFACE, tw=T_SKY_SURFACE,
                    surf_flags=SURF_SKY)
        w.add_brush(bx0-T, by0-T, bz0, bx0, by1+T, bz1,
                    tf=T_SKY_SURFACE, tc=T_SKY_SURFACE, tw=T_SKY_SURFACE,
                    surf_flags=SURF_SKY)
        w.add_brush(bx1, by0-T, bz0, bx1+T, by1+T, bz1,
                    tf=T_SKY_SURFACE, tc=T_SKY_SURFACE, tw=T_SKY_SURFACE,
                    surf_flags=SURF_SKY)
        w.add_brush(bx0, by0-T, bz0, bx1, by0, bz1,
                    tf=T_SKY_SURFACE, tc=T_SKY_SURFACE, tw=T_SKY_SURFACE,
                    surf_flags=SURF_SKY)
        w.add_brush(bx0, by1, bz0, bx1, by1+T, bz1,
                    tf=T_SKY_SURFACE, tc=T_SKY_SURFACE, tw=T_SKY_SURFACE,
                    surf_flags=SURF_SKY)

    def _emit_kill_plane(self):
        """Add an out-of-bounds death volume below the playable layout."""
        w = self.writer
        if not self.rooms:
            return

        all_x = [r.wx for r in self.rooms] + [r.wx + r.w for r in self.rooms]
        all_y = [r.wy for r in self.rooms] + [r.wy + r.d for r in self.rooms]
        min_floor_z = min(r.floor_z for r in self.rooms)
        top_z = min_floor_z - KILL_PLANE_DROP
        bottom_z = min_floor_z - 256 + WALL_T
        if bottom_z >= top_z:
            bottom_z = top_z - 128

        bx0 = min(all_x) - KILL_PLANE_MARGIN
        bx1 = max(all_x) + KILL_PLANE_MARGIN
        by0 = min(all_y) - KILL_PLANE_MARGIN
        by1 = max(all_y) + KILL_PLANE_MARGIN
        brush = w.make_box_brush(
            bx0, by0, bottom_z, bx1, by1, top_z,
            tf=T_TRIM, tc=T_TRIM, tw=T_TRIM,
        )
        w.add_brush_entity("trigger_hurt", {
            "dmg": str(KILL_PLANE_DAMAGE),
            "spawnflags": str(KILL_PLANE_SPAWNFLAGS),
        }, [brush])

        # Visible terrain ground under the whole layout. Was SURF_NODRAW,
        # which rendered as hall-of-mirrors smears from any vantage that can
        # see under the terraces; a rock floor spanning the full skybox
        # footprint terminates every downward sightline on solid ground.
        catch_top_z = top_z - 32
        catch_bottom_z = catch_top_z - WALL_T * 4
        gx0 = min(all_x) - (256 + WALL_T * 4)   # match skybox footprint
        gx1 = max(all_x) + (256 + WALL_T * 4)
        gy0 = min(all_y) - (256 + WALL_T * 4)
        gy1 = max(all_y) + (256 + WALL_T * 4)
        w.add_brush(
            gx0, gy0, catch_bottom_z, gx1, gy1, catch_top_z,
            tf="e1u1/rocks16_2", tc="e1u1/rocks16_2", tw="e1u1/rocks16_2",
            contents=CONTENTS_SOLID,
            value=KILL_PLANE_SURFACE_VALUE,
        )

    def _emit_lethal_drop_guards(self) -> None:
        """Emit the planned solid containment around all void-facing floors."""
        for wall in self.lethal_guard_walls:
            self.writer.add_brush(
                wall.x0, wall.y0, wall.z0, wall.x1, wall.y1, wall.z1,
                tf=self.pal["trim"], tc=self.pal["trim"], tw=self.pal["wall"],
            )
        self.writer.world_props.update({
            "_ml_safety_version": "1",
            "_ml_lethal_edges": str(len(self.lethal_edges)),
            "_ml_lethal_guard_walls": str(len(self.lethal_guard_walls)),
            "_ml_lethal_guard_height": str(LETHAL_GUARD_HEIGHT),
        })

    # ----- top-level generate -----

    def generate(self, grid_n: int = 5) -> Tuple[MapWriter, List[HookZone]]:
        self.build_layout(grid_n)

        self._emit_skybox()
        self._emit_kill_plane()

        for room in self.rooms:
            self._emit_room(room)
        self._emit_lethal_drop_guards()
        for conn in self.connections:
            self._emit_connection(conn)

        self._emit_hallways()
        self._emit_large_buildings()

        self._emit_stairs()
        self._emit_towers()
        self._emit_lane_walls()
        self._emit_lava_pools()
        self._emit_arena_cover()
        # Corners follow the ambient structures so towers, lane walls, lava
        # rims, and cover cannot occupy a pocket promised as enterable. The
        # later objective pass respects the same reserved interior volume.
        self._emit_corner_pockets()

        # Objective geometry and its pickup are selected after every other
        # static blocker is known.  This preserves the objective-shaped tower
        # while preventing later cover, lanes, stairs, or hazards from making
        # the reserved pickup origin startsolid in the compiled BSP.
        self._place_objectives()

        unsafe_surfaces = unsafe_horizontal_sandwiches(
            self.horizontal_surfaces
        )
        if unsafe_surfaces:
            first, second, gap = unsafe_surfaces[0]
            raise RuntimeError(
                "generated unsafe horizontal sandwich: "
                f"{first.surface_id}/{second.surface_id} gap={gap}"
            )

        self._place_entities()
        self._assert_final_route_endpoints_standing_clear()
        self._emit_floor_lighting()
        self._emit_interior_lighting()
        self._annotate_hook_zones()

        return self.writer, self.hook_zones

    def stats(self) -> dict:
        levels = sorted({r.floor_z for r in self.rooms})
        ceiling_bands = {
            "low": sum(1 for r in self.rooms if r.ceil_z - r.floor_z < 256),
            "mid": sum(1 for r in self.rooms
                       if 256 <= r.ceil_z - r.floor_z < 384),
            "high": sum(1 for r in self.rooms if r.ceil_z - r.floor_z >= 384),
        }
        return {
            "style": self.style,
            "palette": self.palette_name,
            "rooms": len(self.rooms),
            "connections": len(self.connections),
            "platforms": sum(len(r.platforms) for r in self.rooms),
            "hook_zones": len(self.hook_zones),
            "hook_claim_candidates_v4_count": len(self.hook_claim_candidates_v4),
            "hook_required": sum(1 for z in self.hook_zones if z.flags & HOOK_REQUIRED),
            "arenas": sum(1 for r in self.rooms if r.kind == 'arena'),
            "kill_planes": 1 if self.rooms else 0,
            "lethal_edges": len(self.lethal_edges),
            "lethal_guard_walls": len(self.lethal_guard_walls),
            "terrace_levels": len(levels),
            "max_elevation": max(levels) if levels else 0,
            "stairs": self.stair_count,
            "towers": self.tower_count,
            "lane_walls": self.lane_wall_count,
            "cover": self.cover_count,
            "hallways": self.hallway_count,
            "corner_pockets": self.corner_count,
            "corners": self.corner_count + 4 * self.large_building_count,
            "large_buildings": self.large_building_count,
            "ceiling_bands": ceiling_bands,
            "lava_pools": len(self.lava_pools),
            "lava_area": sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in self.lava_pools),
            "spawns": len(self.spawn_points),
            "light_regions": len(self.light_regions),
            "floor_lights": len(self.light_sources),
            "interior_light_zones": len(self.interior_light_zones),
            "interior_lights": len(self.interior_light_sources),
            "enterable_room_zones": sum(
                zone.kind in ("arena", "room", "hallway")
                for zone in self.interior_light_zones
            ),
            "enterable_under_platforms": sum(
                zone.kind == "under_platform"
                for zone in self.interior_light_zones
            ),
            "under_platform_lights": sum(
                source.kind == "under_platform" for source in self.light_sources
            ),
            "min_light_coverage": (
                round(min(self.light_coverages.values()), 4)
                if self.light_coverages else 0.0
            ),
            "objectives": len(self.objectives),
            "heat_placed_items": len(self._heat_placed),
            "structure_loot_sites": len(self._loot_sites),
            "armor_total": sum(1 for c, *_ in self._placed_loot + self._heat_placed
                               if c in ARMOR_CLASSES),
            "relax_moves": self.relax_moves,
            "observed_heat_sources": len(self._observed_heat),
        }

    def lighting_manifest(self) -> dict:
        """Serializable inputs for independent static coverage validation."""
        return {
            "version": 2,
            "region_size": LIGHT_REGION_SIZE,
            "sample_spacing": LIGHT_SAMPLE_SPACING,
            "minimum_coverage": MIN_FLOOR_LIGHT_COVERAGE,
            "minimum_light_value": MIN_FLOOR_LIGHT_VALUE,
            "minimum_interior_light_value": MIN_INTERIOR_LIGHT_VALUE,
            "minimum_interior_light_radius": MIN_INTERIOR_LIGHT_RADIUS,
            "spawn_eye_offset": SPAWN_LIGHT_EYE_OFFSET,
            "spawn_eye_samples": [
                list(sample) for sample in self.spawn_eye_samples
            ],
            "player_hull": {
                "width": PLAYER_DIAMETER,
                "standing_height": PLAYER_H,
                "minimum_safe_headroom": MIN_SAFE_HEADROOM,
                "minimum_sandwich_overlap": MIN_SANDWICH_OVERLAP,
                "spawn_escape_distance": SPAWN_ESCAPE_DISTANCE,
            },
            "regions": [
                {
                    "id": region.region_id,
                    "bounds": list(region.bounds),
                    "floor_z": region.floor_z,
                    "samples": [list(sample) for sample in region.samples],
                    "generated_coverage": round(
                        self.light_coverages.get(region.region_id, 0.0), 6
                    ),
                }
                for region in self.light_regions
            ],
            "occluders": [
                [box.x0, box.y0, box.z0, box.x1, box.y1, box.z1]
                for box in self.light_occluders
            ],
            "interior_zones": [
                {
                    "id": zone.zone_id,
                    "kind": zone.kind,
                    "bounds": list(zone.bounds),
                    "floor_z": zone.floor_z,
                    "ceiling_z": zone.ceiling_z,
                    "anchor": list(zone.anchor),
                }
                for zone in self.interior_light_zones
            ],
        }

    def safety_manifest(self) -> dict:
        """Serializable lethal-drop containment contract for validation."""
        return {
            "version": 1,
            "guard_height": LETHAL_GUARD_HEIGHT,
            "guard_thickness": LETHAL_GUARD_THICKNESS,
            "lethal_edges": self.lethal_edges,
            "guard_walls": [
                [wall.x0, wall.y0, wall.z0, wall.x1, wall.y1, wall.z1]
                for wall in self.lethal_guard_walls
            ],
        }

    def hook_claim_candidates_v4_manifest(self) -> dict:
        """Return non-admissible proposals for exact compiled-world replay."""

        return {
            "schema": "q2-hook-claim-candidates-v4",
            "tick_msec": HOOK_RELEASE_TICK_MSEC,
            "status": "unproven",
            "bundle_admissible": False,
            "records": [
                {
                    "claim_id": claim_id,
                    "source_milliunits": list(self._hook_milliunits(candidate.source)),
                    "trace_target_milliunits": list(
                        self._hook_milliunits(candidate.trace_target)
                    ),
                    "landing_milliunits": list(self._hook_milliunits(candidate.landing)),
                    "release_after_ticks": candidate.release_after_ticks,
                    "distance_milliunits": candidate.distance_milliunits,
                    "flags": candidate.flags,
                }
                for claim_id, candidate in self.hook_claim_candidates_v4
            ],
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def generate_map(name: str, seed: Optional[int], out_dir: Path, grid_n: int = 5,
                 style: str = "mixed", observed_heat: Optional[Path] = None,
                 gym: bool = False):
    gen = MapGenerator(seed=seed, style=style, gym=gym)
    if observed_heat and Path(observed_heat).exists():
        # Play-telemetry feedback: deposits recorded from live matches
        # ({"deposits": [{channel,x,y,z,amount,radius}, ...]}). Where real
        # fights happened reinforces; dead zones lose pull — the next
        # generation is shaped by how the last one actually played.
        data = json.loads(Path(observed_heat).read_text())
        gen._observed_heat = list(data.get("deposits", []))
    writer, zones = gen.generate(grid_n=grid_n)

    map_path  = out_dir / f"{name}.map"
    json_path = out_dir / f"{name}.json"
    meta_path = out_dir / f"{name}.meta.json"

    # Prove the complete four-route contract in memory before publishing the
    # first member file. A route refusal must leave no partial source member.
    try:
        from maps.routes import build_route_graph
    except ImportError:
        from routes import build_route_graph   # when run as a script from maps/
    all_items = gen._placed_loot + gen._heat_placed
    objective_items = [
        (o["item"], o["x"], o["y"], o["z"]) for o in gen.objectives
    ]
    route_graph = build_route_graph(
        rooms=gen.rooms, connections=gen.connections,
        items=all_items + objective_items, spawns=gen.spawn_points,
        lava_pools=gen.lava_pools,
        standing_blockers=gen.spawn_blockers,
    )

    writer.write(map_path)

    # Map descriptor for curriculum selection / replay analysis. Separate
    # file: the .json sidecar stays in the line format ML_LoadHookZones parses.
    meta = {
        "name": name,
        "seed": seed,
        "generator": "v6",
        **gen.stats(),
        "lighting": gen.lighting_manifest(),
        "safety": gen.safety_manifest(),
        "hook_claim_candidates_v4": gen.hook_claim_candidates_v4_manifest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # Lattice prior seed: value sites + danger volumes the spatial-memory
    # system can preload, so the bot starts with the map sense the
    # generator already has (objectives = opportunity, lava = threat).
    lattice_path = out_dir / f"{name}.lattice.json"
    lattice = {
        "cell_size": 256,
        "objectives": gen.objectives,
        "danger": [[b.x0, b.y0, b.z0, b.x1, b.y1, b.z1] for b in gen.lava_pools],
        "spawns": [{"x": x, "y": y, "z": z} for x, y, z in gen.spawn_points],
        "items": [{"class": c, "x": x, "y": y, "z": z}
                  for c, x, y, z in all_items],
    }
    lattice_path.write_text(json.dumps(lattice, indent=2) + "\n")

    # Route-graph sidecar: item/spawn nodes, risk-weighted edges, archetype
    # routes (offense/survival/control/balanced) with respawn periods. The
    # substrate for runtime item-timing + buffed-intercept route choice.
    (out_dir / f"{name}.routes.json").write_text(
        json.dumps(route_graph, indent=1) + "\n")

    # Runtime-compatible projection: simple text, one unique geometry per line.
    # This raw generator output is not bundle-admissible. Compiled preflight
    # replaces it with exactly six proven rows before canonical claim binding.
    # anchor_x anchor_y anchor_z  landing_x landing_y landing_z  distance  flags
    lines = ["# q2-ml-bot hook zones — unproven generator projection"]
    lines.append("# bundle_admissible: false; compiled preflight must materialize final rows")
    lines.append(f"# generated: {name}  zones: {len(zones)}")
    lines.append("# anchor.x anchor.y anchor.z  landing.x landing.y landing.z  distance  flags")
    for z in zones:
        a = z.anchor; l = z.landing
        lines.append(f"{a[0]:.1f} {a[1]:.1f} {a[2]:.1f}  "
                     f"{l[0]:.1f} {l[1]:.1f} {l[2]:.1f}  "
                     f"{z.distance:.1f} {z.flags}")
    json_path.write_text("\n".join(lines) + "\n")

    s = gen.stats()
    print(f"  {name} [{s['style']}/{s['palette']}]: {s['rooms']} rooms "
          f"({s['arenas']} arenas), {s['terrace_levels']} terraces, "
          f"{s['stairs']} stairs, {s['towers']} towers, "
          f"{s['lane_walls']} lane walls, {s['lava_pools']} lava pools, "
          f"{s['platforms']} platforms, "
          f"{s['hallways']} halls, {s['corners']} corners, "
          f"{s['large_buildings']} buildings, "
          f"{s['hook_zones']} hook zones ({s['hook_required']} required)")
    return map_path, json_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",   default=None,   help="map name for a single generated map")
    p.add_argument("--prefix", default="mltrain", help="batch map prefix")
    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--count",  type=int, default=1, help="generate N maps")
    p.add_argument("--grid",   type=int, default=5, help="grid size (default 5)")
    p.add_argument("--style",  default="mixed", choices=STYLES,
                   help="map style (mixed = random per map)")
    p.add_argument("--observed-heat", default=None,
                   help="JSON of play-telemetry heat deposits to seed the field")
    p.add_argument("--outdir", default=str(Path(__file__).parent / "generated"),
                   help="output directory")
    args = p.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.count == 1 else [
        (args.seed or 0) + i for i in range(args.count)
    ]

    for i, seed in enumerate(seeds):
        if seed is None:
            import time; seed = int(time.time()) + i
        name = args.name if (args.name and args.count == 1) else f"{args.prefix}_{seed:08d}"
        generate_map(name, seed, out_dir, grid_n=args.grid, style=args.style,
                     observed_heat=args.observed_heat)

    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
