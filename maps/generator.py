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
from typing import List, Optional, Tuple

# ── Q2 world constants ────────────────────────────────────────────────────────

GRID_SIZE   = 512          # world units per grid cell
WALL_T      = 16           # brush wall thickness
PLAYER_H    = 56           # player standing height
PLAYER_XY_HALF = 16        # Quake II player bbox half-width
PLAYER_MINS_Z = -24        # origin-relative standing bbox bottom
PLAYER_MAXS_Z = 32         # origin-relative standing bbox top
JUMP_H      = 50           # max jump height (units)
HOOK_MIN    = 150          # minimum useful hook distance
HOOK_MAX    = 580          # maximum hook range
DM_SPAWN_COUNT = 8         # headroom for six-player live matches
MIN_SPAWN_SEPARATION = 384 # minimum 2D spacing for generated DM spawns
SPAWN_EDGE_MARGIN = 96     # keep starts away from room edges/sky walls
SPAWN_SOLID_MARGIN = 48    # extra XY clearance from cover and low blockers
KILL_PLANE_DROP = 64       # distance below the lowest playable floor
KILL_PLANE_MARGIN = 512    # XY overhang beyond the generated layout
KILL_PLANE_DAMAGE = 100000 # instant-kill damage for out-of-bounds falls
KILL_PLANE_SPAWNFLAGS = 12 # trigger_hurt: SILENT | NO_PROTECTION
KILL_PLANE_SURFACE_VALUE = 31337 # hook-impact marker; must match g_local.h

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
LAVA_DEPTH   = 8
LAVA_MIN, LAVA_MAX = 224, 384   # pool edge length

STYLES = ("open", "towers", "canyon", "pits", "mixed")

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


@dataclass
class SolidBox:
    x0: int; y0: int; z0: int
    x1: int; y1: int; z1: int


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
        lines.append('"light" "100"')
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
        self.spawn_points: List[Tuple[int, int, int]] = []
        self.spawn_blockers: List[SolidBox] = []
        self.writer = MapWriter()
        self._adjacent: List[Tuple[int, int]] = []   # grid-neighbour room pairs
        self.lava_pools: List[SolidBox] = []
        self.objectives: List[dict] = []             # lattice-seeded value sites
        self._placed_loot: List[Tuple[str, int, int, int]] = []  # heat emitters
        self._heat_placed: List[Tuple[str, int, int, int]] = []
        self._loot_sites: List[Tuple[str, int, int, int]] = []  # (kind, x, y, z)
        self._observed_heat: List[dict] = []   # play-telemetry deposits
        self.relax_moves = 0
        self.tower_count = 0
        self.stair_count = 0
        self.lane_wall_count = 0
        self.cover_count = 0

        # Resolve style → feature knobs (mixed rolls per map)
        r = self.rng
        if style == "mixed":
            style = r.choice(("open", "towers", "canyon", "pits"))
        self.style = style
        self.palette_name = r.choice(list(PALETTES.keys()))
        self.pal = PALETTES[self.palette_name]
        self.terrace_levels = {"open": 1, "towers": 2, "canyon": 1, "pits": 3}[style]
        self.tower_prob     = {"open": 0.15, "towers": 0.9, "canyon": 0.2, "pits": 0.3}[style]
        # Judge: zero lane walls leaves verticality carrying all the flow —
        # every style keeps some chance of engineered chokepoints.
        self.lane_prob      = {"open": 0.35, "towers": 0.35, "canyon": 0.9, "pits": 0.3}[style]
        self.lava_prob      = {"open": 0.1, "towers": 0.15, "canyon": 0.15, "pits": 0.6}[style]

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
            # corridors are 1 cell wide, 1-2 long — orientation handled in placement
            w = GRID_SIZE
            d = GRID_SIZE
            h = r.randint(128, 192)
        return w, d, h

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
        while frontier and len(occupied) < int(N*N * 0.55):
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
            if dist_centre <= 1 and len(occupied) > 5:
                kind = 'arena'
            elif rng.random() < 0.35:
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

        # 4. Add platforms to arenas and tall rooms
        for room in self.rooms:
            if room.kind in ('arena', 'room'):
                self._add_platforms(room)

        # 5. Register every floor plate as a spawn blocker: with terracing,
        # a higher room's plate can overhang a lower room's area, and spawns
        # or items placed there would be inside solid rock. Boundary contact
        # (standing ON a plate) does not count as overlap.
        for room in self.rooms:
            self.spawn_blockers.append(SolidBox(
                room.wx, room.wy, -WALL_T,
                room.wx + room.w, room.wy + room.d, room.floor_z))
            # Ceilings too: a low corridor ceiling can overhang a higher
            # neighbouring terrace at head height. (-8 also covers light strips.)
            self.spawn_blockers.append(SolidBox(
                room.wx, room.wy, room.ceil_z - 8,
                room.wx + room.w, room.wy + room.d, room.ceil_z + WALL_T))

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

    def _add_platforms(self, room: Room):
        """Add floating platforms inside tall rooms."""
        rng = self.rng
        room_h = room.ceil_z - room.floor_z
        if room_h < 256:
            return
        n_platforms = rng.randint(1, 2) if room.kind == 'arena' else 1
        for _ in range(n_platforms):
            plat_z  = room.floor_z + rng.randint(128, room_h - 96)
            plat_w  = rng.randint(96, min(256, room.w - 64))
            plat_d  = rng.randint(96, min(256, room.d - 64))
            plat_x0 = room.wx + rng.randint(32, max(33, room.w - plat_w - 32))
            plat_y0 = room.wy + rng.randint(32, max(33, room.d - plat_d - 32))
            room.platforms.append({
                'z': plat_z, 'thick': 16,
                'x0': plat_x0, 'y0': plat_y0,
                'x1': plat_x0 + plat_w, 'y1': plat_y0 + plat_d,
            })
            # With terraces, a platform from one room can overhang another
            # room's floor at body height — keep spawns/items out of it.
            # (Covers the railing strip at z-8 too.)
            self.spawn_blockers.append(SolidBox(
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
                        surf_flags=SURF_LIGHT, value=200)

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
            w.add_brush(x0, y0, fz, x1, y1, fz + cover_h,
                        tf=self.pal['metal'], tc=self.pal['metal'], tw=self.pal['trim'])
            self.spawn_blockers.append(SolidBox(x0, y0, fz, x1, y1, fz + cover_h))

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
                    w.add_brush(x0, cy - STAIR_WIDTH // 2, low.floor_z,
                                x1, cy + STAIR_WIDTH // 2, step_top,
                                tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['trim'])
                    self.spawn_blockers.append(SolidBox(
                        x0, cy - STAIR_WIDTH // 2, low.floor_z,
                        x1, cy + STAIR_WIDTH // 2, step_top))
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
                    w.add_brush(cx - STAIR_WIDTH // 2, y0, low.floor_z,
                                cx + STAIR_WIDTH // 2, y1, step_top,
                                tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['trim'])
                    self.spawn_blockers.append(SolidBox(
                        cx - STAIR_WIDTH // 2, y0, low.floor_z,
                        cx + STAIR_WIDTH // 2, y1, step_top))
            self.stair_count += 1

    def _place_objectives(self):
        """Objective-first (lattice-seeded) generation, stage 1.

        Sample a high-value site BEFORE structure generation and build the
        guard geometry around it: the map is shaped by the objective rather
        than the objective scattered onto the map. The site is exported in
        the .lattice.json sidecar so the spatial-memory system can preload
        it as an opportunity prior — the bot spawns knowing value exists
        there, and the geometry controls how it can be reached."""
        rng = self.rng
        w = self.writer
        arenas = sorted((r for r in self.rooms if r.kind == 'arena'),
                        key=lambda r: r.w * r.d, reverse=True)
        if not arenas:
            return
        room = arenas[0]
        th = rng.randint(TOWER_H_MIN + 32, TOWER_H_MAX)
        tx = room.wx + rng.randint(192, max(193, room.w - TOWER_BASE - 192))
        ty = room.wy + rng.randint(192, max(193, room.d - TOWER_BASE - 192))
        fz = room.floor_z
        top = min(fz + th, room.ceil_z - PLAYER_H - 24)
        w.add_brush(tx, ty, fz, tx + TOWER_BASE, ty + TOWER_BASE, top,
                    tf=self.pal['light'], tc=self.pal['metal'], tw=self.pal['metal'])
        self.spawn_blockers.append(SolidBox(tx, ty, fz,
                                            tx + TOWER_BASE, ty + TOWER_BASE, top))
        cx_t = tx + TOWER_BASE // 2
        cy_t = ty + TOWER_BASE // 2
        w.add_entity("item_quad", {"origin": f"{cx_t} {cy_t} {top + 24}"})
        if HOOK_MIN <= (top - fz) <= HOOK_MAX:
            self.hook_zones.append(HookZone(
                anchor=(float(cx_t), float(cy_t), float(top)),
                landing=(float(cx_t), float(cy_t), float(top + PLAYER_H + 4)),
                distance=float(top - fz),
                flags=HOOK_WALL | HOOK_REQUIRED,
            ))
        self.objectives.append({
            "item": "item_quad", "x": cx_t, "y": cy_t, "z": top + 24,
            "guard": "tower", "height": top - fz, "value": 1.0,
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
                w.add_brush(tx, ty, fz, tx + TOWER_BASE, ty + TOWER_BASE, top,
                            tf=self.pal['metal'], tc=self.pal['metal'], tw=self.pal['wall'])
                self.spawn_blockers.append(SolidBox(tx, ty, fz,
                                                    tx + TOWER_BASE, ty + TOWER_BASE, top))
                cx_t = tx + TOWER_BASE // 2
                cy_t = ty + TOWER_BASE // 2
                # Tower-top loot site: the heat pass decides WHAT goes here —
                # weapon repulsion diversifies nearby towers instead of every
                # top getting an identical power-weapon + armour pair.
                self._loot_sites.append(("tower", cx_t, cy_t, top + 24))
                if HOOK_MIN <= th <= HOOK_MAX:
                    self.hook_zones.append(HookZone(
                        anchor=(float(cx_t), float(cy_t), float(top)),
                        landing=(float(cx_t), float(cy_t), float(top + PLAYER_H + 4)),
                        distance=float(th),
                        flags=HOOK_WALL | HOOK_REQUIRED,
                    ))
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
            for side in (-1, 1):
                if along_x:
                    wy = cy + side * room.d // 4
                    seg_x0 = room.wx + 128
                    seg_x1 = room.wx + room.w - 128
                    gap0 = cx - LANE_GAP // 2
                    gap1 = cx + LANE_GAP // 2
                    for x0, x1 in ((seg_x0, gap0), (gap1, seg_x1)):
                        if x1 - x0 < 64:
                            continue
                        w.add_brush(x0, wy - LANE_WALL_T // 2, fz,
                                    x1, wy + LANE_WALL_T // 2, fz + LANE_WALL_H,
                                    tf=self.pal['wall'], tc=self.pal['trim'], tw=self.pal['wall'])
                        self.spawn_blockers.append(SolidBox(
                            x0, wy - LANE_WALL_T // 2, fz,
                            x1, wy + LANE_WALL_T // 2, fz + LANE_WALL_H))
                else:
                    wx_ = cx + side * room.w // 4
                    seg_y0 = room.wy + 128
                    seg_y1 = room.wy + room.d - 128
                    gap0 = cy - LANE_GAP // 2
                    gap1 = cy + LANE_GAP // 2
                    for y0, y1 in ((seg_y0, gap0), (gap1, seg_y1)):
                        if y1 - y0 < 64:
                            continue
                        w.add_brush(wx_ - LANE_WALL_T // 2, y0, fz,
                                    wx_ + LANE_WALL_T // 2, y1, fz + LANE_WALL_H,
                                    tf=self.pal['wall'], tc=self.pal['trim'], tw=self.pal['wall'])
                        self.spawn_blockers.append(SolidBox(
                            wx_ - LANE_WALL_T // 2, y0, fz,
                            wx_ + LANE_WALL_T // 2, y1, fz + LANE_WALL_H))
                self.lane_wall_count += 1

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
            size = rng.randint(LAVA_MIN, LAVA_MAX)
            px = room.wx + rng.randint(96, max(97, room.w - size - 96))
            py = room.wy + rng.randint(96, max(97, room.d - size - 96))
            fz = room.floor_z
            # Pool sits on the floor with a low rim so it reads as terrain
            w.add_brush(px - 16, py - 16, fz, px + size + 16, py + size + 16, fz + 12,
                        tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['trim'])
            w.add_brush(px, py, fz + 12, px + size, py + size, fz + 12 + LAVA_DEPTH,
                        tf=T_LAVA, tc=T_LAVA, tw=T_LAVA,
                        surf_flags=SURF_LIGHT | SURF_WARP, value=120,
                        contents=CONTENTS_LAVA)
            box = SolidBox(px - 16, py - 16, fz, px + size + 16, py + size + 16, fz + 64)
            self.spawn_blockers.append(box)
            self.lava_pools.append(box)
            # Mega health on the rim: worth the burn risk
            mx = px - 48 if rng.random() < 0.5 else px + size + 48
            w.add_entity("item_health_mega",
                         {"origin": f"{mx} {py + size // 2} {fz + 24}"})
            self._placed_loot.append(("item_health_mega", mx, py + size // 2, fz + 24))

    # ----- entity placement -----

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

    def _spawn_is_clear(self, room: Room, x: int, y: int, z: int) -> bool:
        if not self._inside_room_for_spawn(room, x, y):
            return False
        if self._spawn_overlaps_blocker(x, y, z):
            return False
        for sx, sy, _ in self.spawn_points:
            if math.hypot(x - sx, y - sy) < MIN_SPAWN_SEPARATION:
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
            fz_check = fz + 32   # probe above the floor brush, like spawns/items
            target = rng.randint(2, 4)
            placed = 0
            for _ in range(target * 6):
                if placed >= target:
                    break
                # mid-ring: away from dead-center (where objectives sit) and
                # the walls, so it shapes the open fighting area.
                cx = room.wx + rng.randint(room.w // 4, max(room.w // 4 + 1, 3 * room.w // 4))
                cy = room.wy + rng.randint(room.d // 4, max(room.d // 4 + 1, 3 * room.d // 4))
                if self._spawn_overlaps_blocker(cx, cy, fz_check):
                    continue
                box = SolidBox(cx - COVER_W // 2, cy - COVER_W // 2, fz,
                               cx + COVER_W // 2, cy + COVER_W // 2, fz + COVER_H)
                w.add_brush(box.x0, box.y0, box.z0, box.x1, box.y1, box.z1,
                            tf=self.pal['trim'], tc=self.pal['trim'], tw=self.pal['wall'])
                self.spawn_blockers.append(box)
                self.cover_count += 1
                placed += 1

    def _place_combat_spawns(self):
        """Place a validated, well-separated set of deathmatch starts.

        Preferred: a wide ring in a big arena (clean opening sightlines).
        With terracing, that arena may be partly buried under higher
        overlapping plates or the ring may collapse inward — then fall back
        to DISTRIBUTED placement: one spawn per room across the map."""
        arenas = sorted(
            (r for r in self.rooms if r.kind == 'arena'),
            key=lambda r: r.w * r.d, reverse=True,
        )
        for room in arenas:
            placed = self._try_combat_spawns_in(room)
            if placed and self._spawn_span_ok(placed):
                self._emit_spawn_entities(placed)
                return
            if placed:   # ring fit but too tight — undo and go distributed
                self.spawn_points = self.spawn_points[:-len(placed)]
                break

        # Distributed: one spawn per room, greedily choosing the room whose
        # centre is farthest from the spawns placed so far (maximises spread).
        placed = []
        map_cx = sum(r.wx + r.w // 2 for r in self.rooms) // max(1, len(self.rooms))
        map_cy = sum(r.wy + r.d // 2 for r in self.rooms) // max(1, len(self.rooms))
        # Revisit rooms after the first pass: larger layouts can safely hold
        # more than one start per room, while _spawn_is_clear still enforces
        # both geometry clearance and global separation.
        for _ in range(DM_SPAWN_COUNT * 2):
            if len(placed) >= DM_SPAWN_COUNT:
                break
            ranked = sorted(
                self.rooms,
                key=lambda r: (
                    min(
                        math.hypot(r.wx + r.w // 2 - px, r.wy + r.d // 2 - py)
                        for px, py, _, _ in placed
                    ) if placed else float(r.w * r.d)
                ),
                reverse=True,
            )
            progress = False
            for room in ranked:
                cand = self._choose_spawn(room)
                if cand is None:
                    continue
                x, y, fz = cand
                yaw = int(round((math.degrees(math.atan2(map_cy - y, map_cx - x)) + 360.0) % 360.0))
                self.spawn_points.append((x, y, fz))
                placed.append((x, y, fz, yaw))
                progress = True
                if len(placed) >= DM_SPAWN_COUNT:
                    break
            if not progress:
                break
        if len(placed) < DM_SPAWN_COUNT:
            raise RuntimeError(
                f"could not place {DM_SPAWN_COUNT} clear deathmatch spawns"
            )
        self._emit_spawn_entities(placed)

    def _item_spot(self, rng, xc: int, yc: int, fz: int, spread: int):
        """Random item position near (xc, yc) that does not intersect cover
        blocks or other spawn blockers (droptofloor deletes startsolid items).
        Returns None when no clear spot is found."""
        for _ in range(12):
            x = xc + rng.randint(-spread, spread)
            y = yc + rng.randint(-spread, spread)
            if not self._spawn_overlaps_blocker(x, y, fz):
                return x, y
        return None

    def _place_entities(self):
        self._place_combat_spawns()

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
            cands = candidate_sites(TIER_CELL[tier])
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

        cands = candidate_sites(TIER_CELL[3])
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
            if rune_cands:
                cx, cy, cz = rng.choice(rune_cands)   # one central rack site
                offs = [(-48, 0), (48, 0), (0, -48), (0, 48), (0, 0)]
                for name, (dx, dy) in zip(runes, offs):
                    x, y, z = cx + dx, cy + dy, cz
                    t = ITEM_TEMPLATES[name]
                    for ch, amt in t["emit"].items():
                        field.deposit(ch, x, y, z, amt, t["radius"])
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
                pool = rng.sample(sites_cache[stride],
                                  min(48, len(sites_cache[stride])))
                # Relocation honours the same hard floors as placement —
                # without them the relax pass walked armour right back
                # onto the weapon heat it is meant to answer from afar.
                others = [(c, px, py, pz) for c, px, py, pz in
                          self._placed_loot + self._heat_placed
                          if (c, px, py, pz) != self._heat_placed[idx]]
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
        return out

    # ----- hook zone annotation -----

    def _annotate_hook_zones(self):
        """
        For each room+platform combination, identify hook anchor points:
          1. Ceiling above open floor  → HOOK_CEILING
          2. Platform ceiling above floor gap → HOOK_CEILING | HOOK_REQUIRED
             (gap is defined as platform height > JUMP_H from floor)
        """
        for room in self.rooms:
            xc = room.wx + room.w // 2
            yc = room.wy + room.d // 2
            fz = room.floor_z
            cz = room.ceil_z
            room_h = cz - fz

            # Ceiling anchor for the main room floor
            anchor_z = cz - WALL_T
            if HOOK_MIN <= anchor_z - fz <= HOOK_MAX:
                self.hook_zones.append(HookZone(
                    anchor=(float(xc), float(yc), float(anchor_z)),
                    landing=(float(xc), float(yc), float(fz + PLAYER_H)),
                    distance=float(anchor_z - fz),
                    flags=HOOK_CEILING,
                ))

            # Platform ceiling anchors
            for p in room.platforms:
                px = (p['x0'] + p['x1']) // 2
                py = (p['y0'] + p['y1']) // 2
                pz = p['z']
                dist_from_floor = pz - fz

                # Anchor on underside of platform
                under_z = pz
                if HOOK_MIN <= dist_from_floor <= HOOK_MAX:
                    flags = HOOK_CEILING
                    if dist_from_floor > JUMP_H + 16:
                        flags |= HOOK_REQUIRED  # can't reach by jumping
                    self.hook_zones.append(HookZone(
                        anchor=(float(px), float(py), float(under_z)),
                        landing=(float(px), float(py), float(pz + PLAYER_H + 4)),
                        distance=float(dist_from_floor),
                        flags=flags,
                    ))

                # Anchor on main ceiling above platform top
                above_z = cz - WALL_T
                dist_from_plat = above_z - (pz + p['thick'])
                if HOOK_MIN <= dist_from_plat <= HOOK_MAX:
                    self.hook_zones.append(HookZone(
                        anchor=(float(px), float(py), float(above_z)),
                        landing=(float(px), float(py), float(pz + p['thick'] + PLAYER_H)),
                        distance=float(dist_from_plat),
                        flags=HOOK_CEILING,
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

    # ----- top-level generate -----

    def generate(self, grid_n: int = 5) -> Tuple[MapWriter, List[HookZone]]:
        self.build_layout(grid_n)

        self._emit_skybox()
        self._emit_kill_plane()

        for room in self.rooms:
            self._emit_room(room)
        for conn in self.connections:
            self._emit_connection(conn)

        self._place_objectives()
        self._emit_stairs()
        self._emit_towers()
        self._emit_lane_walls()
        self._emit_lava_pools()
        self._emit_arena_cover()

        self._place_entities()
        self._annotate_hook_zones()

        return self.writer, self.hook_zones

    def stats(self) -> dict:
        levels = sorted({r.floor_z for r in self.rooms})
        return {
            "style": self.style,
            "palette": self.palette_name,
            "rooms": len(self.rooms),
            "connections": len(self.connections),
            "platforms": sum(len(r.platforms) for r in self.rooms),
            "hook_zones": len(self.hook_zones),
            "hook_required": sum(1 for z in self.hook_zones if z.flags & HOOK_REQUIRED),
            "arenas": sum(1 for r in self.rooms if r.kind == 'arena'),
            "kill_planes": 1 if self.rooms else 0,
            "terrace_levels": len(levels),
            "max_elevation": max(levels) if levels else 0,
            "stairs": self.stair_count,
            "towers": self.tower_count,
            "lane_walls": self.lane_wall_count,
            "cover": self.cover_count,
            "lava_pools": len(self.lava_pools),
            "lava_area": sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in self.lava_pools),
            "spawns": len(self.spawn_points),
            "objectives": len(self.objectives),
            "heat_placed_items": len(self._heat_placed),
            "structure_loot_sites": len(self._loot_sites),
            "armor_total": sum(1 for c, *_ in self._placed_loot + self._heat_placed
                               if c in ARMOR_CLASSES),
            "relax_moves": self.relax_moves,
            "observed_heat_sources": len(self._observed_heat),
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

    writer.write(map_path)

    # Map descriptor for curriculum selection / replay analysis. Separate
    # file: the .json sidecar stays in the line format ML_LoadHookZones parses.
    meta = {"name": name, "seed": seed, "generator": "v3", **gen.stats()}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # Lattice prior seed: value sites + danger volumes the spatial-memory
    # system can preload, so the bot starts with the map sense the
    # generator already has (objectives = opportunity, lava = threat).
    lattice_path = out_dir / f"{name}.lattice.json"
    all_items = gen._placed_loot + gen._heat_placed
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
    try:
        from maps.routes import build_route_graph
    except ImportError:
        from routes import build_route_graph   # when run as a script from maps/
    objective_items = [(o["item"], o["x"], o["y"], o["z"]) for o in gen.objectives]
    route_graph = build_route_graph(
        rooms=gen.rooms, connections=gen.connections,
        items=all_items + objective_items, spawns=gen.spawn_points,
        lava_pools=gen.lava_pools)
    (out_dir / f"{name}.routes.json").write_text(
        json.dumps(route_graph, indent=1) + "\n")

    # Sidecar format: simple text, one zone per line, easy to parse in C
    # anchor_x anchor_y anchor_z  landing_x landing_y landing_z  distance  flags
    lines = ["# q2-ml-bot hook zones — one per line"]
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
