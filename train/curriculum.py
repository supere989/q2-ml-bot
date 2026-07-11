#!/usr/bin/env python3
"""
curriculum.py — entropy-guarded curriculum evolution (pool-level churn).

PRINCIPLE (held-out finding 2026-06-17, see memory heldout-generalization-findings):
observation data is SAFE to absorb into the training DISTRIBUTION but DANGEROUS
to absorb into the policy INPUT. Enriching the policy's per-map observation
(ext-obs) caused overfitting — B collapsed to 0.56 on unseen maps while leaner
C held 1.17. So this loop deliberately does NOT re-place items toward observed
fight clusters (tools/evolve_maps.py does that; it converges a single map and
re-creates the overfitting surface at the curriculum level).

Instead it churns the POOL:
  * RETIRE maps the (exploratory) policy has MASTERED — high training K/D means
    no gradient left; a solved map only invites overfitting if it lingers.
  * REPLACE each with a FRESH-seed map (full entropy / new geometry), validated
    for structural coherence before it enters the pool.
  * KEEP hard maps (still teaching).
Net effect: a moving-target curriculum that sustains generalization pressure —
absorb-for-entropy, never absorb-for-convergence.

SAFETY / production guards:
  * Gated by Q2_CURRICULUM_EVOLVE=1 — only the pilot run instantiates this.
  * Isolated by map_prefix (e.g. mlcur_) so it never touches maps other runs
    glob (mltrain_*). Fresh seeds live in a disjoint seed space.
  * Compiles in a BACKGROUND process — training never blocks on qbsp.
  * Capped churn per cycle + a pool floor so the distribution can't lurch.
  * A map only swaps in after compile success + structural validation; the
    map it replaces stays live until then (pool size is preserved).
"""

import itertools
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_BSP_BYTES = 30_000          # a real compiled arena; rejects empty/failed builds
FRESH_SEED_BASE = 90_000_000    # disjoint from mltrain (52xx), held (9xxx), gym


def map_combat_summary(servers):
    """Aggregate (kills, deaths, samples) per map from live session memories.

    samples = kills+deaths, a play-volume proxy so we don't judge a map the
    policy has barely touched. Training-time K/D is the RIGHT signal here: it
    measures 'is there gradient left on this map', not checkpoint quality.
    """
    agg = {}
    for srv in servers:
        for sr in getattr(srv, "_spatial_rewards", []):
            for mp, cells in getattr(sr, "session_memories", {}).items():
                k = sum(float(c.kills) for c in cells.values())
                d = sum(float(c.deaths) for c in cells.values())
                ck, cd, cs = agg.get(mp, (0.0, 0.0, 0.0))
                agg[mp] = (ck + k, cd + d, cs + k + d)
    return agg


class CurriculumEvolver:
    def __init__(self, *, map_prefix, out_dir=None, q2maps=None,
                 max_churn=1, mastery_kd=2.5, min_samples=40.0, min_pool=8,
                 log=print):
        self.prefix     = map_prefix
        self.out_dir    = Path(out_dir or (ROOT / "maps" / "generated"))
        self.q2maps     = Path(q2maps or os.environ.get(
            "Q2_MAPS_DIR", "/home/raymond/q2_lithium_merge/baseq2/maps"))
        self.max_churn  = int(max_churn)
        self.mastery_kd = float(mastery_kd)
        self.min_samples= float(min_samples)
        self.min_pool   = int(min_pool)
        self.log        = log
        self.pending    = {}     # fresh_name -> {"proc","bsp","replaces"}
        self.retired    = set()  # mastered maps already churned out
        self._used_seeds = set()

    # -- seed allocation -----------------------------------------------------
    def _fresh_seed(self, pool):
        for nm in list(pool) + list(self.retired) + list(self.pending):
            m = re.search(r"_(\d{8})$", nm)
            if m:
                self._used_seeds.add(int(m.group(1)))
        for s in itertools.count(FRESH_SEED_BASE):
            if s not in self._used_seeds:
                self._used_seeds.add(s)
                return s

    # -- background generate + compile --------------------------------------
    def _launch(self, replaces, pool):
        seed = self._fresh_seed(pool)
        name = f"{self.prefix}_{seed:08d}"
        py = (
            "import sys; sys.path.insert(0, %r); "
            "from pathlib import Path; "
            "from maps.generator import generate_map; "
            "generate_map(%r, %d, Path(%r), style='mixed')"
            % (str(ROOT), name, seed, str(self.out_dir))
        )
        cmd = ("python3 -c \"%s\" && MAP_PREFIX=%s bash %s/maps/compile.sh %s/%s.map"
               % (py, self.prefix, ROOT, self.out_dir, name))
        proc = subprocess.Popen(["bash", "-lc", cmd],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
        self.pending[name] = {"proc": proc, "bsp": self.q2maps / f"{name}.bsp",
                              "replaces": replaces}
        self.log(f"  [curriculum] forging {name} (seed {seed}) to replace "
                 f"mastered {replaces}")

    def _valid(self, name, bsp):
        if not (bsp.exists() and bsp.stat().st_size >= MIN_BSP_BYTES):
            return False
        # structural sanity from the generator's own meta sidecar if present
        meta = self.out_dir / f"{name}.json"
        try:
            if meta.exists():
                import json
                m = json.loads(meta.read_text())
                if m.get("rooms", 1) < 1 or m.get("spawns", m.get("spawn_points", 4)) < 4:
                    return False
        except Exception:
            pass
        return True

    def _harvest(self, servers):
        for name, info in list(self.pending.items()):
            if info["proc"].poll() is None:
                continue
            self.pending.pop(name)
            if info["proc"].returncode == 0 and self._valid(name, info["bsp"]):
                drop = info["replaces"]
                for srv in servers:
                    srv.map_pool = [m for m in srv.map_pool if m != drop] + \
                                   ([name] if name not in srv.map_pool else [])
                self.retired.add(drop)
                self.log(f"  [curriculum] SWAP +{name} -{drop}  "
                         f"(pool={len(servers[0].map_pool)})")
            else:
                self.log(f"  [curriculum] {name} failed compile/validate — "
                         f"discarded; {info['replaces']} stays")

    # -- the per-cycle entry point ------------------------------------------
    def step(self, servers, total_env_steps):
        if not servers:
            return
        self._harvest(servers)
        pool = list(servers[0].map_pool)
        # effective size after in-flight retirements complete
        effective = len(pool) - len(self.pending)
        if effective <= self.min_pool:
            return
        kd = map_combat_summary(servers)
        pending_targets = {i["replaces"] for i in self.pending.values()}
        cand = []
        for mp in pool:
            if mp in self.retired or mp in pending_targets:
                continue
            k, d, s = kd.get(mp, (0.0, 0.0, 0.0))
            if s < self.min_samples:
                continue
            ratio = k / max(d, 1.0)
            if ratio >= self.mastery_kd:
                cand.append((ratio, s, mp))
        cand.sort(reverse=True)
        slots = min(self.max_churn, effective - self.min_pool)
        for _, _, mp in cand[:slots]:
            self._launch(mp, pool)
