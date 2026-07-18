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
mod guide;
mod l0;
mod manifest;
mod objective;
mod recovery;
mod runtime;
mod storage;

pub use admission::{
    B1_RUNTIME_AUTHORITY_SEAL_SCHEMA, B1AuthorityExecutables, B1AuthorityIdentities,
    B1AuthorityIdentity, B1NormativeDocuments, B1RuntimeAuthoritySeal, COLLISION_ORACLE_NAME,
    COLLISION_ORACLE_SCHEMA, CollisionOracleAdmission, CollisionParameters, CollisionSourceClosure,
    EdgeAdmission, FALL_ORACLE_NAME, FALL_ORACLE_SCHEMA, FallOracleAdmission, FallParameters,
    FallSourceClosure, HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA, HOOK_PARITY_CASES_V1,
    HOOK_PARITY_NAME, HOOK_PARITY_SCHEMA, HookOracleAdmission, HookParameters,
    HookParityAttestation, HookSourceClosure, MASK_PLAYERSOLID_V1, MASK_SHOT_V1,
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
pub use guide::{
    GUIDE_CANDIDATE_LIMIT, GUIDE_CANDIDATE_WIDTH, GUIDE_COST_WORLD_SCALE,
    GUIDE_DIRECTION_WORLD_SCALE, GUIDE_FEATURE_NAMES, GUIDE_FEATURE_WIDTH, GuideFeatureBlock,
    ObjectiveClass,
};
pub use l0::{L0_CELLS_PER_CHUNK, L0_CHUNK_SIDE, L0BitPlane, L0Chunk, L0ScalarPlane, SparseL0};
pub use manifest::{
    ArtifactManifest, AtlasCounts, AtlasManifest, BspIdentity, ChannelManifest, GridManifest,
    HullManifest, ManifestBudgets, RecoveryPhysicsManifest, ToolIdentity, sha256_hex,
};
pub use objective::{
    AtlasObjective, AtlasObjectives, DEFAULT_AVAILABILITY_BELIEF, OBJECTIVE_ARTIFACT_SUFFIX,
    OBJECTIVE_CLASS_COUNT, OBJECTIVE_LIMIT, OBJECTIVE_MEDIA_TYPE, OBJECTIVE_SCHEMA,
    OBJECTIVE_TARGET_MAX_DISTANCE, ObjectiveBelief,
};
pub use recovery::{
    HAZARD_CLEARANCE_BOUNDARY_Q8, HAZARD_CLEARANCE_UNREACHABLE_HAZARD,
    HAZARD_CLEARANCE_UNREACHABLE_SAFE, HOOK_RECOVERY_WALK_BUDGET_TICKS, HazardComponentField,
    HookNecessityEvidence, PolicyHazardBits, RECOVERY_CLEARANCE_WORLD_SCALE,
    RECOVERY_COST_WORLD_SCALE, RECOVERY_EVIDENCE_FIELD_NAMES, RECOVERY_EVIDENCE_SCHEMA,
    RECOVERY_FEATURE_NAMES, RECOVERY_FEATURE_WIDTH, RECOVERY_GAME_TICK_HZ,
    RECOVERY_REPAIR_L2_RADIUS, RECOVERY_REPAIR_NODE_LIMIT, RECOVERY_TTI_SCALE_SECONDS,
    RECOVERY_WALK_DISTANCE_Q8_PER_TICK, RECOVERY_WALK_SPEED_Q8_PER_SECOND, RecoveryEvidence,
    RecoveryFeatureBlock, RecoveryOverlay, RecoveryQuery, evaluate_hook_necessity,
    install_static_costs, install_static_hazard_clearances, recovery_features, solve_static_costs,
    solve_static_hazard_clearances, validate_static_costs, validate_static_hazard_clearances,
};
pub use runtime::{
    ADVISORY_SPATIAL_WIDTH, ATLAS_MEDIA_TYPE, AtlasQueryCountersSnapshot, AtlasQueryTiming,
    AtlasRuntime, AtlasRuntimeSlot, TimedAdvisoryFeatures, advisory_spatial_feature_names,
};
pub use storage::{
    ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, AtlasAggregateCell, AtlasArtifact, AtlasLimits,
    ENVELOPE_MAGIC, decode_zstd_envelope, encode_zstd_envelope,
};
