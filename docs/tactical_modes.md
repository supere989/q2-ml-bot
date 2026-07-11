# Tactical mode taxonomy

> **Status note (2026-07-10):** this taxonomy assumes two load-bearing
> capabilities that live-deployment testing found are not yet solid —
> reliable aim/target-tracking (needed by Offense/Hunt/FINISH HIM/Ambush)
> and correctly-signed lattice pull (needed by Explore/Control/Defense; the
> "engagement" plume is currently pulling the wrong direction). See
> `q2-ml-bot/README.md` § Known Issues before building on this taxonomy —
> the emergent/reward-weight implementation fork below will inherit
> whatever the underlying plumes actually do.

The bot's behavioral intents. A **mode is a weighting over the belief plumes** —
"in this situation, which plumes pull and which repel" — so modes reuse the same
lattice machinery as item-timing; they are not a separate system. The modes fall
out of regions in the state space already defined by the health axis, exchange
ratio, enemy-belief confidence, and info-staleness.

| Mode | Triggers when… | Reads (plumes) | Does |
|---|---|---|---|
| **Explore** (Discovery Traversal) | belief stale / quiet / no contact | confidence gradient, readiness | traverse to refresh the table, time items, find enemies — productive idle |
| **Control** (Greed) | healthy, no threat, items ripening | readiness (all axes), denial-value | run the item loop, lock the mega/quad cycle, starve the enemy economy |
| **Offense** | enemy present, even/favorable trade, geared | enemy-belief, kill-opportunity, weapon-ground | press the fight you're in |
| **Hunt** (Seek) | enemy suspected not seen, I have the edge | enemy-belief, predicted-intercept, audio | pursue toward where they'll be |
| **FINISH HIM** | enemy confirmed weak/fleeing | enemy-belief (sharp), intercept, cut-off | commit hard, cut escape, deny their heal |
| **Ambush** | their route overlaps mine (item-absence), I have position | enemy-take inference, intercept, cover | hold concealed on the contested item's route, strike |
| **Defense** | I hold valuable ground / economy | control/territory, sightline, cover | cover the approaches, area-deny |
| **Survival** | low health + threat | survival readiness, cover/escape, threat | break to recovery + cover, refuse trades |
| **Reposition** (Gear-up) | even/losing trade but not desperate; or under-armed | readiness (needed buff), cover, enemy-belief | the chicken **swerve** — disengage down a buffed-intercept route, re-arm, re-commit |

**Deferred (advanced):** *Bait / Feint* — deliberately show yourself to pull the
enemy onto your ambush or into a hazard. Models the enemy modeling *you*; shelve
until the plumes are stable.

## Notes on boundaries (why these are distinct)

- **Hunt vs Offense vs Finish** — Hunt finds the fight (enemy not seen), Offense
  wins the fight you're in, Finish executes a confirmed weak target.
- **Control vs Explore vs Defense** — Explore *reduces uncertainty*, Control
  *exploits known item timing* (holds the clock, not the ground), Defense *holds
  ground*. Control is the highest-level arena-FPS win condition.
- **Reposition vs Survival** — Survival is health-desperate (recover or die);
  Reposition is a calm tactical break to grab the buff that flips the matchup.
  Without Reposition as its own mode, "back off and gear up" has nowhere to live.

## Implementation fork

- **Emergent** (start here): modes are reward-weight profiles the bot drifts
  between; no explicit mode variable. The plumes + state axes already produce
  mode-like behavior.
- **Explicit mode head** (add once plumes are in): a learned latent the policy
  switches (hierarchical-RL / options), so we can *see* which mode it's in and
  *reward correct mode selection*.
