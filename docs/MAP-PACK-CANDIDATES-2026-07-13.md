# External map-pack candidates — 2026-07-13

This is a research shortlist, not an installation manifest. No external map
was downloaded, added to the public fast-download server, or admitted to PPO.

## Classic-lane first intake

1. `spirit2dm3`: designed for 3–6 players and explicitly tested with 3ZB2.
   The archive includes BSP/MAP/LOC/source-license material and custom sounds.
   The author's work is GPL, but every bundled sound still needs an asset-level
   redistribution audit before public rehosting.
2. `WDDM05`: 2–8 players; a close match for the six-player live lane.
3. `WDDM03`: 2–16 players; broader arena/courtyard impressions.
4. `WDDM06`: 2–24 players; useful geometry diversity, but likely needs a
   density gate before a six-player rotation.

Upstream references:

- Spirit collection: <https://maps.rcmd.org/quake2/>
- `spirit2dm3` readme: <https://maps.rcmd.org/quake2/spirit2dm3/spirit2dm3.txt>
- Dervish WDDM packages: <https://dondeq2.com/2017/10/10/the-dervish-depository-quake-2-maps/>

## Hold or separate lane

- Q2Cafe Contest 10 documents twenty original-engine DM maps, but the original
  bundle and contributor-specific redistribution grants were not recovered:
  <https://celephais.net/board/view_thread.php?id=61188>
- The SIN Net pack lists 75 classic multiplayer BSPs, but its compilation page
  does not provide a sufficient redistribution grant. It is a private-candidate
  corpus only until every author notice is recovered:
  <https://www.markshan.com/thesinraven/the_sin_net_multiplayer_map_pack.htm>
- Map-Center DM Jam 2023 contains nine polished maps with bot support and an
  intact-pack redistribution grant. It targets Quake II Remaster, so it belongs
  in a separate QBSP/BSPX-compatible canary rather than the current legacy
  Yamagi/Lithium lane:
  <https://www.map-center.com/resources/>
  <https://www.moddb.com/games/quake-2/addons/quake-2-remaster-deathmatch-map-jam-2023>
- Notorious T.O.P. is an unusual 2–6-player office impression and permits free
  redistribution only as an intact archive with its readme. Its author warns
  that gameplay and r_speeds were secondary, so it is an ablation candidate,
  not a primary rotation map:
  <https://www.geocities.ws/rick_barkhouse/quake2/readme.html>

## Admission contract

1. Quarantine the original archive. Reject executables, DLLs, symlinks, path
   traversal, duplicate-case paths, and unreasonable decompression ratios.
2. Record source URL, archive SHA-256, BSP SHA-256, author, license/readme, and
   the provenance of every texture, sound, model, and sky.
3. Require classic `IBSP` version 38 for the current lane. QBSP/BSPX/ReRelease
   maps go to a separate Remaster-compatible lane.
4. Resolve every asset path case-sensitively and run static geometry checks.
   Require at least eight geometry-clear, well-separated DM origins before
   using a map for six-player training impressions.
5. Canary the exact public q2ded/Lithium module through repeated map changes,
   six real network clients, at least 1,000 respawns, and missing-asset/entity
   log checks.
6. Run a frozen policy and measure encounter rate, kills, stuck time,
   jump/hook dependence, illumination, spawn occupancy, and lattice coverage.
7. Tag every impression with source map ID and BSP hash. A restricted or weak
   map must remain removable from both training data and rotation history.
8. Public fast-download requires an explicit grant covering the complete
   package and every custom asset. Otherwise retain only the upstream URL and
   do not rehost the map.
