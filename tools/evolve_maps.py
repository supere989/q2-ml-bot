#!/usr/bin/env python3
"""
evolve_maps.py — close the generation loop.

For every observed-heat file exported by training (observed_heat/<map>.heat.json),
regenerate that map with the SAME seed (identical geometry, identical style
roll) but with the observed play heat seeding the placement field: the item
economy re-organises around where the fights actually happened. Output goes
to maps/generated/ ready for compile.sh.

Usage:
    python3 tools/evolve_maps.py                  # evolve all maps with heat data
    python3 tools/evolve_maps.py --map mltrain_00005103
    python3 tools/evolve_maps.py --min-deposits 25   # skip thinly-played maps
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maps.generator import generate_map  # noqa: E402

HEAT_DIR = ROOT / "observed_heat"
OUT_DIR = ROOT / "maps" / "generated"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default=None, help="evolve a single map by name")
    p.add_argument("--heat-dir", default=str(HEAT_DIR))
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--min-deposits", type=int, default=10,
                   help="skip maps with fewer observed deposits")
    args = p.parse_args()

    heat_dir = Path(args.heat_dir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(heat_dir.glob(f"{args.map or '*'}.heat.json"))
    if not files:
        print(f"no heat files in {heat_dir}")
        return 1

    evolved = 0
    for f in files:
        name = f.name.replace(".heat.json", "")
        m = re.search(r"_(\d{8})$", name)
        if not m:
            print(f"  skip {name}: cannot derive seed from name")
            continue
        seed = int(m.group(1))
        data = json.loads(f.read_text())
        n_dep = len(data.get("deposits", []))
        if n_dep < args.min_deposits:
            print(f"  skip {name}: only {n_dep} deposits")
            continue
        # Same seed + style=mixed reproduces the identical geometry and
        # style roll; only the placement field differs.
        generate_map(name, seed, out_dir, style="mixed", observed_heat=f)
        evolved += 1

    print(f"evolved {evolved}/{len(files)} maps → {out_dir}")
    print("compile with: bash maps/compile.sh maps/generated/<name>.map")
    return 0


if __name__ == "__main__":
    sys.exit(main())
