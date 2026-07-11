#!/usr/bin/env python3
"""item_timing.py — per-bot item-timing belief state + readiness heat plume.

Timing is DISCOVERED, not precomputed. Each item spawn point gets a row that
the bot fills in by observation: when it last saw the item present, when it
went away, and — using Quake-2's static respawn periods — when it predicts the
item back. The table's output is not numbers the policy reads but a READINESS
HEAT PLUME projected into the lattice: each item radiates a gradient that ramps
as its respawn approaches, peaks when available, and collapses when consumed.
The bot senses the plumes through the same opportunity channel it already uses,
so "which powerup is up, and when, relative to the others" becomes a heat
field it drifts through rather than a schedule it computes.

Contention adds variance: an item found gone that the table predicted present
(and the bot did not take) is an interrupt — confidence drops, the timer
re-phases to worst-case, and the plume smears. That same asymmetry is the
enemy-presence signal the absence-inference layer reads.

The static skeleton (positions, respawn periods, tactical value) comes from the
route-graph sidecar (<map>.routes.json). Discovery is fed by engine item
perception (visible spawns: present/absent) once that wire field lands; until
then the table still advances on the bot's own pickups + predicted respawns.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ItemRow:
    spawn_id: int
    cls: str
    pos: Tuple[float, float, float]
    respawn_period: float           # static Q2 seconds; 0 = event-driven (runes)
    effectiveness: float            # tactical value weight
    axis: str = "value"             # offense | survival | value

    present: bool = True            # current belief
    last_seen_present_t: float = 0.0
    became_present_t: float = 0.0   # when it last (re)appeared — for despawn timing
    present_lifetime: float = math.inf   # despawns if present-and-untouched this long
    last_consumed_t: float = -1.0   # last present->absent transition (consume OR despawn)
    taken_by_me: bool = False       # last consumption was this bot's pickup
    despawned: bool = False         # last vacate was a timeout, not a pickup
    predicted_available_t: float = 0.0
    confidence: float = 1.0         # 1=sharp clock, →0 = contested/uncertain
    contested: bool = False         # last disappearance was an unexplained pickup (enemy)

    def update_prediction(self):
        if self.respawn_period <= 0.0 or self.last_consumed_t < 0.0:
            self.predicted_available_t = self.last_seen_present_t
        else:
            self.predicted_available_t = self.last_consumed_t + self.respawn_period

    def readiness(self, t: float) -> float:
        """0 just after consumption → 1 at/after predicted respawn. If believed
        present, 1.0. Event-driven items (period 0) read present-or-unknown."""
        if self.present:
            return 1.0
        if self.respawn_period <= 0.0 or self.last_consumed_t < 0.0:
            return 0.5  # unknown phase — mild standing pull
        frac = (t - self.last_consumed_t) / self.respawn_period
        return max(0.0, min(1.0, frac))


class ItemTimingTable:
    """One per ML bot. Reset at map load (the t=0 baseline where every item is
    known present)."""

    # readiness ramps into a plume this far out (s) so the bot pre-positions.
    LOOKAHEAD_S = 6.0
    BASE_RADIUS = 256.0
    # Rune anti-camp negative window as a fraction of respawn period, sized so
    # the post-grab negative integral ≈ the presence pull → net-neutral voxel.
    RUNE_ANTICAMP_FRAC = 0.65   # tuned: rune cycle heat integral ≈ -0.8 (≈ neutral, never net-positive)

    def __init__(self):
        self.rows: Dict[int, ItemRow] = {}
        self.t0_set = False
        self.last_enemy_take: Optional[dict] = None  # most recent inferred steal

    # ---- lifecycle ----------------------------------------------------------
    def reset_from_routes(self, route_graph: dict, t: float):
        """Seed the table from the route-graph nodes — map-start baseline where
        every item is present and the clock is zeroed."""
        self.rows.clear()
        self.last_enemy_take = None
        for n in route_graph.get("nodes", []):
            if n.get("type") != "item":
                continue
            cls = n["class"]
            # Placed runes despawn-and-respawn on their own (engine
            # PLACED_RUNE_PRESENT); everything else only leaves on pickup.
            lifetime = 35.0 if cls.startswith("rune_") else math.inf
            self.rows[n["id"]] = ItemRow(
                spawn_id=n["id"], cls=cls,
                pos=(n["x"], n["y"], n["z"]),
                respawn_period=float(n.get("respawn_s", 30.0)),
                effectiveness=float(n.get("value", 0.3)),
                axis=n.get("axis", "value"),
                present=True, last_seen_present_t=t, became_present_t=t,
                present_lifetime=lifetime, confidence=1.0,
            )
        self.t0_set = True

    # ---- discovery ----------------------------------------------------------
    def observe(self, t: float, seen_present_ids, seen_absent_ids,
                my_pickup_ids=None):
        """Feed one frame of perception.

        seen_present_ids / seen_absent_ids: spawn ids the bot can currently see
        and whether the item is there. my_pickup_ids: spawn ids the bot itself
        consumed this frame.
        """
        my_pickup_ids = set(my_pickup_ids or ())
        for sid in (my_pickup_ids):
            row = self.rows.get(sid)
            if row and row.present:
                self._vacate(row, t, by_me=True, despawn=False)

        for sid in seen_present_ids:
            row = self.rows.get(sid)
            if not row:
                continue
            if not row.present:
                # reappeared on schedule (or earlier than feared) — sharpen
                row.confidence = min(1.0, row.confidence + 0.25)
                row.became_present_t = t
            row.present = True
            row.last_seen_present_t = t
            row.contested = False

        for sid in seen_absent_ids:
            row = self.rows.get(sid)
            if not row or sid in my_pickup_ids:
                continue
            if row.present:
                # Gone and not by my hand. Distinguish the two exits: if it had
                # been present past its lifetime it timed out (despawn — no
                # enemy); otherwise someone grabbed it (consume → infer enemy).
                despawn = (t - row.became_present_t) >= row.present_lifetime
                self._vacate(row, t, by_me=False, despawn=despawn)

        # Unobserved despawn: a rune present in belief past its lifetime has
        # timed out even if we didn't see the empty spot — vacate it so the
        # voxel presence heat stops claiming a rune that's no longer there.
        for row in self.rows.values():
            if (row.present and row.present_lifetime < math.inf
                    and (t - row.became_present_t) >= row.present_lifetime):
                self._vacate(row, t, by_me=False, despawn=True)

        for row in self.rows.values():
            row.update_prediction()

    def _vacate(self, row: ItemRow, t: float, by_me: bool, despawn: bool):
        """Rune/item left the voxel — by pickup (consume) or timeout (despawn).
        Both adjust the voxel presence heat (via last_consumed_t); only a
        consume-by-other implies enemy presence."""
        row.present = False
        row.last_consumed_t = t       # last present->absent transition (any cause)
        row.taken_by_me = by_me
        row.despawned = despawn
        if despawn:
            row.contested = False     # timed out — no enemy, timing stays sharp
        elif by_me:
            row.confidence = 1.0
            row.contested = False
        else:
            row.confidence = max(0.2, row.confidence * 0.5)  # smeared timer
            row.contested = True
            self.last_enemy_take = {
                "spawn_id": row.spawn_id, "cls": row.cls,
                "pos": row.pos, "t": t, "axis": row.axis,
                "effectiveness": row.effectiveness,
            }
        row.update_prediction()

    # ---- readiness plume ----------------------------------------------------
    def readiness_deposits(self, t: float, want_axis: Optional[str] = None
                           ) -> List[dict]:
        """Heat deposits the bot senses. want_axis (from the health/exchange
        state) up-weights the axis the bot currently needs; None = neutral."""
        out = []
        for row in self.rows.values():
            is_rune = row.cls.startswith("rune_")

            # Runes are placed (for exposure) but we must NOT let the bot
            # pattern-learn their fixed spawn points — that's positional
            # overfit, exactly what the ephemeral design avoided. So a
            # just-emptied rune voxel emits NEGATIVE heat (push away from the
            # spot) that fades over the respawn, while presence/ripening emit
            # the usual positive. The voxel's heat then TIME-AVERAGES to ~0:
            # the bot chases a rune that's actually there, but never memorizes
            # where runes come from. (Per design: counter the +placement heat
            # with -heat in the same voxel.)
            if (is_rune and not row.present and row.respawn_period > 0
                    and row.last_consumed_t >= 0):
                eta = row.predicted_available_t - t
                if eta > self.LOOKAHEAD_S:
                    # Negative only during an early post-grab window, sized so
                    # its integral ≈ the presence pull's → the voxel nets to
                    # ~0 over a cycle (no positional memory), then goes silent
                    # until it ripens back into the positive pre-position pull.
                    age = t - row.last_consumed_t
                    win = self.RUNE_ANTICAMP_FRAC * row.respawn_period
                    if age < win:
                        neg = -row.effectiveness * row.confidence * (1.0 - age / win)
                        radius = self.BASE_RADIUS * (1.0 + (1.0 - row.confidence))
                        out.append({"channel": "readiness", "x": row.pos[0],
                                    "y": row.pos[1], "z": row.pos[2],
                                    "amount": round(neg, 3),
                                    "radius": round(radius, 1), "axis": row.axis,
                                    "contested": row.contested, "anti_camp": True})
                    continue

            r = row.readiness(t)
            if r <= 0.01:
                continue
            # pre-position: items ripening within LOOKAHEAD get a partial pull
            if not row.present and row.respawn_period > 0 and row.last_consumed_t >= 0:
                eta = row.predicted_available_t - t
                if eta > self.LOOKAHEAD_S:
                    r *= max(0.0, 1.0 - (eta - self.LOOKAHEAD_S) / self.LOOKAHEAD_S)
            axis_w = 1.0
            if want_axis and row.axis == want_axis:
                axis_w = 1.6
            elif want_axis and row.axis == "value":
                axis_w = 1.2  # vampire-class bridges always somewhat wanted
            amount = r * row.effectiveness * row.confidence * axis_w
            if amount <= 0.01:
                continue
            # low confidence smears the plume: wider, so the gradient is fuzzy
            radius = self.BASE_RADIUS * (1.0 + (1.0 - row.confidence))
            out.append({"channel": "readiness", "x": row.pos[0], "y": row.pos[1],
                        "z": row.pos[2], "amount": round(amount, 3),
                        "radius": round(radius, 1), "axis": row.axis,
                        "contested": row.contested})
        return out

    # ---- enemy inference (feeds #28) ---------------------------------------
    def recent_enemy_take(self, t: float, window_s: float = 4.0) -> Optional[dict]:
        e = self.last_enemy_take
        if e and (t - e["t"]) <= window_s:
            return e
        return None
