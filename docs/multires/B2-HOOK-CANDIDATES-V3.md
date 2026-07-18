# B2 generated hook candidates and materialization v3

Status: retired historical evidence only

V3 reused its rounded compiled collision endpoint as the next replay's trace
target. CM collision epsilon made that transform non-idempotent and caused a
sealed 71428 hook record to reproduce a different landing. The sole operational
contract is `B2-HOOK-CANDIDATES-V4.md`. No current parser, materializer,
analyzer, campaign, bundle, or runtime path accepts or converts V3 artifacts.
