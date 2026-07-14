# B1-C corpus quarantine and IBSP-38 metadata contract

Status: implemented on `agent/b1-corpus-quarantine` for B1 integration.

This lane admits authored Quake II maps for analysis without extracting or
executing archive content. It records source and license provenance, detects
exact copies and coordinate-free near-copy candidates, and parses the
non-collision metadata needed by later Atlas stages. It does **not** derive
collision, passability, hookability, coordinates, graphs, cells, routes, or
training labels.

## Trust boundary

`harness.corpus_quarantine.quarantine_archive()` is the only general incoming
archive lane. It recognizes ZIP/PK3-by-magic and Quake PAK containers and fails
closed before BSP parsing when any configured bound or content rule is
violated. It never extracts members and never invokes member content.

The default bounds are:

| Limit | Default |
|---|---:|
| Archive bytes | 512 MiB |
| Total uncompressed bytes | 1 GiB |
| One member | 128 MiB |
| Members | 8,192 |
| Selected BSP members | 256 |
| Compression ratio | 100:1 |
| UTF-8 member path | 512 bytes |

All totals are checked from container metadata and each member is then read as
a bounded stream. ZIP reads are forced through end-of-stream so CRC validation
occurs. PAK directory and member ranges must be in bounds and non-overlapping.
The scanner rejects traversal, absolute/drive paths, empty or dot components,
NULs, Unicode-normalized case collisions, symlinks, encrypted ZIP members,
nested archives, executable/script suffixes, and common executable/script
magic. Configuration and scripts are active content and are not admitted.

`inventory_stock_pak()` is a deliberately narrower fixture lane for an already
installed retail PAK. It validates the whole PAK directory and ranges but reads
only the exact names `maps/q2dm1.bsp` through `maps/q2dm8.bsp`. It cannot select
community maps or arbitrary members. The resulting archive and BSP hashes must
still match provenance before admission. This exception exists because retail
containers can contain engine configuration and legacy internal paths that
must remain forbidden in the general untrusted archive lane.

## BSP metadata boundary

`harness.ibsp38.parse_ibsp38()` accepts only `IBSP` version 38. Every lump must
be within the file, fixed-record lumps must be integral, and non-empty lump
ranges must not overlap. The parser enforces configurable file, lump, entity,
token, string, model, face, cluster, and visibility-decompression bounds.

The following metadata is exposed:

- entity key/value records and class counts;
- deathmatch/start spawn counts, item classes, movers, triggers, teleports, and
  brush-submodel references;
- model bounds, origin, headnode, and face ranges, including required model 0;
- face count, texture names, light styles/offsets, and lightmapped-face count;
- light-data byte count and SHA-256;
- PVS/PHS cluster count, row bytes, compressed/decompressed byte counts, and
  aggregate visible-cluster statistics.

References used by this metadata are range checked, including model faces,
entity `*N` models, face planes/texinfo/surfedges/light offsets, node and leaf
references, leaf face/brush ranges, texinfo animation links, and PVS/PHS row
offsets and RLE expansion. No collision interpretation is performed.

## Provenance and admission

Every selected BSP requires one `q2-corpus-provenance-v1` record containing:

- a stable lower-case canonical ID and optional lower-case aliases;
- exactly one source URL or manual/local origin;
- author, license name, and human-verifiable license evidence;
- `analysis-only` or `redistributable` status;
- the quarantine archive SHA-256 and selected BSP SHA-256.

Missing, duplicate, ambiguous, or hash-mismatched records reject the complete
admission. `analysis-only` is the safe default when no redistribution grant is
recorded. Admission does not copy BSP bytes into the repository.

## Duplicate classification

Equal BSP SHA-256 values are exact copies and receive one canonical ID. Other
pairs are scored for manual near-copy review using only entity/item class
multisets and ratios of entity, model, face, light-data, and visibility-cluster
counts. The default review threshold is `0.94`. The signature intentionally
contains no entity origins, model bounds, coordinates, topology, or graph data;
it cannot become an accidental spatial prior.

Near-copy classification is advisory. It does not merge or discard a map
without review and cannot establish license compatibility.

## Stock fixture gate

The initial corpus is the locally installed retail set q2dm1–q2dm8 only:

- `docs/multires/stock-q2dm1-q2dm8.provenance.json` pins archive/BSP hashes and
  records every map as analysis-only.
- `tests/fixtures/corpus/stock-q2dm1-q2dm8.json` pins the structural inventory.
- `tests/test_corpus_quarantine.py` reparses the local PAK when present and
  compares every map with both records. Absence skips only this local-data test;
  synthetic malformed/archive/security tests remain mandatory.

Community ingestion remains disabled until this structural gate passes for all
eight stock maps and a separately reviewed source/license record is available.

## CLI output

`python tools/corpus_quarantine.py ARCHIVE --output REPORT.json` emits stable,
sorted JSON containing quarantine metadata, provenance-bound corpus entries,
coordinate-free duplicate signatures/classifications, and (for
`--stock-q2dm`) the stock inventory. Raw entity key/value records are omitted
from CLI reports; only the validated entity catalogs are emitted. CLI output is
analysis data; it does not extract, install, deploy, or start services.
