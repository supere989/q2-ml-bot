# Generated Map Lighting and Overhead Geometry Contract

Generator v5 makes lighting, overhead clearance, and spawn escape properties
machine-checkable in both the `.map` source and `.meta.json` sidecar.

## Player geometry thresholds

The Quake II standing hull is 32 units wide and 56 units high. A horizontal
slab pair is rejected when all of the following are true:

- the overhead/platform/roof footprints overlap by at least 48 by 48 units
  (the hull plus 8 units of lateral margin on every side);
- the free gap between the lower slab top and upper slab bottom is at least 56
  units, so a standing player can enter it;
- the free gap is less than 96 units, the minimum safe movement headroom.

Touching slab assemblies are not gaps. The validator treats a platform and its
8-unit underside trim as one assembly, and treats a ceiling light panel and
the ceiling brush touching it as one assembly. The generator adjusts unsafe
overlapping ceiling bands deterministically, rejects unsafe platform/building
roof candidates, and aborts generation if any registered pair remains.

Every deathmatch spawn must have a clear 32-unit-wide column from its floor to
96 units above the floor. It must also have floor support and collision-free
headroom along at least one 96-unit horizontal escape path, sampled every 16
units in eight compass directions. `tools/validate_maps.py` recomputes these
properties from emitted brush AABBs rather than trusting generator counters.

## Lighting thresholds

Spawn-clear base floors remain partitioned into deterministic 512-unit regions
with 128-unit probes plus exact spawn probes. Version 2 of the lighting
contract raises the minimum direct coverage from 90% to 98% and requires every
counted source to have a `light` value of at least 650. Generated overhead
sources use value 900 and a tagged 448-unit radius; under-platform/building
fills use value 700. World ambient light is 180, and emissive arena ceiling
panels use surface value 300.

Every enterable room-like location owns a dedicated internal point light:
arenas, ordinary rooms, hallways, corner pockets, building interiors, and
under-platform spaces with at least 96 units of headroom. Those sources are
tagged with `_ml_interior_light`, `_ml_zone`, `_ml_kind`, and `_ml_radius`.
They use value 850 and radius 384; validation requires at least value 800 and
radius 320, an origin inside the zone below its ceiling/roof, and a direct path
to the zone's enterability-proven anchor. Light above an occluding roof or
platform cannot satisfy this requirement.

## Validation and compiled evidence

Static promotion fails for a weakened/missing lighting contract, missing or
weak tagged sources, any interior zone without its own direct source, ambient
light below 180, floor coverage below 98%, an unsafe horizontal sandwich, a
blocked spawn column, or a spawn without an escape path.

When a same-name `.bsp` is present beside the `.map`, validation also reads the
Quake II BSP lighting lump. The qrad evidence must be non-empty and no larger
than `MAX_MAP_LIGHTING` (2,097,152 bytes). A corrupt header, empty lightdata,
or lightdata overflow fails static promotion. Source-only validation reports
the compiled evidence as unavailable rather than pretending qrad ran.
