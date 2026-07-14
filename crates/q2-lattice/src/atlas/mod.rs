//! Deterministic, map-static multi-resolution Atlas schema and storage.
//!
//! Collision and movement facts enter this module only after validation by the
//! pinned engine oracles. Runtime code reads these records; it does not infer
//! collision from BSP bytes.

mod admission;
mod aggregate;
mod coord;
mod error;
mod graph;
mod l0;
mod manifest;
mod storage;

pub use admission::{
    COLLISION_ORACLE_NAME, COLLISION_ORACLE_SCHEMA, CollisionOracleAdmission, CollisionParameters,
    CollisionSourceClosure, EdgeAdmission, HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA,
    HOOK_PARITY_CASES_V1, HOOK_PARITY_NAME, HOOK_PARITY_SCHEMA, HookOracleAdmission,
    HookParameters, HookParityAttestation, HookSourceClosure, MASK_PLAYERSOLID_V1, MASK_SHOT_V1,
    ORACLE_SEMANTIC_VERSION, OracleAdmissions, OracleBspBinding, OracleToolIdentity,
    PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA, PmoveOracleAdmission, PmoveParameters,
    PmoveSourceClosure,
};
pub use aggregate::{ConservativeChild, CorridorWitness, StaticAggregate, aggregate_conservative};
pub use coord::{ATLAS_CELL_SIZES, AtlasLevel, AtlasOrigin, GridIndex, L0Address};
pub use error::{AtlasError, AtlasResult};
pub use graph::{
    COST_INFINITY, EdgeInput, EdgeRecord, EdgeType, L1Graph, L1Node, NodeFlags, Stance,
};
pub use l0::{L0_CELLS_PER_CHUNK, L0_CHUNK_SIDE, L0BitPlane, L0Chunk, L0ScalarPlane, SparseL0};
pub use manifest::{
    ArtifactManifest, AtlasCounts, AtlasManifest, BspIdentity, ChannelManifest, GridManifest,
    HullManifest, ManifestBudgets, ToolIdentity, sha256_hex,
};
pub use storage::{
    ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, AtlasAggregateCell, AtlasArtifact, AtlasLimits,
    ENVELOPE_MAGIC, decode_zstd_envelope, encode_zstd_envelope,
};
