use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::guide::{GUIDE_FEATURE_NAMES, GUIDE_FEATURE_WIDTH, GuideFeatureBlock};
use super::objective::{AtlasObjectives, OBJECTIVE_MEDIA_TYPE, ObjectiveBelief, ObjectiveGuide};
use super::recovery::{
    HazardComponentField, RECOVERY_FEATURE_NAMES, RECOVERY_FEATURE_WIDTH, RecoveryFeatureBlock,
    RecoveryQuery, recovery_features_at, resolve_recovery_node, validate_static_costs,
    validate_static_hazard_clearances,
};
use super::{
    ATLAS_MAGIC, AtlasArtifact, AtlasError, AtlasLimits, AtlasManifest, AtlasResult,
    ENVELOPE_MAGIC, decode_zstd_envelope, sha256_hex,
};

pub const ATLAS_MEDIA_TYPE: &str = "application/vnd.q2.atlas-v1";
pub const ADVISORY_SPATIAL_WIDTH: usize = RECOVERY_FEATURE_WIDTH + GUIDE_FEATURE_WIDTH;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AtlasQueryTiming {
    pub atlas_lookup_ns: u64,
    pub recovery_ns: u64,
    pub guide_ns: u64,
    pub total_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedAdvisoryFeatures {
    pub values: [f32; ADVISORY_SPATIAL_WIDTH],
    pub timing: AtlasQueryTiming,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AtlasQueryCountersSnapshot {
    pub accepted_queries: u64,
    pub atlas_lookup_ns: u64,
    pub recovery_ns: u64,
    pub guide_ns: u64,
    pub total_ns: u64,
}

#[derive(Debug, Default)]
struct AtlasQueryCounters {
    accepted_queries: AtomicU64,
    atlas_lookup_ns: AtomicU64,
    recovery_ns: AtomicU64,
    guide_ns: AtomicU64,
    total_ns: AtomicU64,
}

impl AtlasQueryCounters {
    fn record(&self, timing: AtlasQueryTiming) {
        self.accepted_queries.fetch_add(1, Ordering::Relaxed);
        self.atlas_lookup_ns
            .fetch_add(timing.atlas_lookup_ns, Ordering::Relaxed);
        self.recovery_ns
            .fetch_add(timing.recovery_ns, Ordering::Relaxed);
        self.guide_ns.fetch_add(timing.guide_ns, Ordering::Relaxed);
        self.total_ns.fetch_add(timing.total_ns, Ordering::Relaxed);
    }

    fn snapshot(&self) -> AtlasQueryCountersSnapshot {
        AtlasQueryCountersSnapshot {
            accepted_queries: self.accepted_queries.load(Ordering::Relaxed),
            atlas_lookup_ns: self.atlas_lookup_ns.load(Ordering::Relaxed),
            recovery_ns: self.recovery_ns.load(Ordering::Relaxed),
            guide_ns: self.guide_ns.load(Ordering::Relaxed),
            total_ns: self.total_ns.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AtlasRuntime {
    manifest: AtlasManifest,
    artifact: Arc<AtlasArtifact>,
    objectives: Arc<AtlasObjectives>,
    objective_guide: Arc<ObjectiveGuide>,
    hazard_components: Arc<HazardComponentField>,
    artifact_name: String,
    objective_artifact_name: String,
    manifest_sha256: String,
    atlas_sha256: String,
    objective_sha256: String,
    map_epoch: u64,
    counters: Arc<AtlasQueryCounters>,
}

impl AtlasRuntime {
    /// Admit one complete, map-epoch-fenced Atlas. There is deliberately no
    /// missing-Atlas or legacy-lattice fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        manifest_bytes: &[u8],
        artifact_name: &str,
        atlas_transport: &[u8],
        objective_artifact_name: &str,
        objective_payload: &[u8],
        bsp_bytes: &[u8],
        expected_map_id: &str,
        map_epoch: u64,
        limits: &AtlasLimits,
    ) -> AtlasResult<Self> {
        if expected_map_id.is_empty()
            || artifact_name.is_empty()
            || objective_artifact_name.is_empty()
        {
            return Err(AtlasError::InvalidFormat(
                "runtime Atlas identity is empty".to_owned(),
            ));
        }
        let manifest = AtlasManifest::from_canonical_json(manifest_bytes, limits)?;
        if manifest.bsp.canonical_map_id != expected_map_id {
            return Err(AtlasError::InvalidFormat(format!(
                "runtime map {expected_map_id} != Atlas map {}",
                manifest.bsp.canonical_map_id
            )));
        }
        if bsp_bytes.len() as u64 != manifest.bsp.size_bytes
            || sha256_hex(bsp_bytes) != manifest.bsp.sha256
        {
            return Err(AtlasError::DigestMismatch);
        }
        let atlas_artifacts: Vec<_> = manifest
            .artifacts
            .iter()
            .filter(|(_, artifact)| artifact.media_type == ATLAS_MEDIA_TYPE)
            .collect();
        if atlas_artifacts.len() != 1 || atlas_artifacts[0].0 != artifact_name {
            return Err(AtlasError::InvalidFormat(
                "runtime manifest must name exactly one Atlas artifact".to_owned(),
            ));
        }
        let objective_artifacts: Vec<_> = manifest
            .artifacts
            .iter()
            .filter(|(_, artifact)| artifact.media_type == OBJECTIVE_MEDIA_TYPE)
            .collect();
        if objective_artifacts.len() != 1 || objective_artifacts[0].0 != objective_artifact_name {
            return Err(AtlasError::InvalidFormat(
                "runtime manifest must name exactly one objective artifact".to_owned(),
            ));
        }
        let identity = manifest.artifacts.get(artifact_name).ok_or_else(|| {
            AtlasError::InvalidFormat(format!(
                "runtime manifest has no Atlas artifact {artifact_name}"
            ))
        })?;
        if identity.media_type != ATLAS_MEDIA_TYPE {
            return Err(AtlasError::InvalidFormat(format!(
                "runtime artifact {artifact_name} has media type {}",
                identity.media_type
            )));
        }
        let raw = if atlas_transport.starts_with(ATLAS_MAGIC) {
            atlas_transport.to_vec()
        } else if atlas_transport.starts_with(ENVELOPE_MAGIC) {
            if atlas_transport.len() as u64 != identity.compressed_size {
                return Err(AtlasError::DigestMismatch);
            }
            decode_zstd_envelope(atlas_transport, limits)?
        } else {
            return Err(AtlasError::InvalidFormat(
                "runtime Atlas transport has no admitted magic".to_owned(),
            ));
        };
        let artifact = manifest.decode_and_verify_atlas_artifact(artifact_name, &raw, limits)?;
        validate_static_costs(&artifact.l1)?;
        validate_static_hazard_clearances(&artifact.l1)?;
        let atlas_sha256 = sha256_hex(&raw);
        let objective_identity =
            manifest
                .artifacts
                .get(objective_artifact_name)
                .ok_or_else(|| {
                    AtlasError::InvalidFormat(format!(
                        "runtime manifest has no objective artifact {objective_artifact_name}"
                    ))
                })?;
        if objective_identity.media_type != OBJECTIVE_MEDIA_TYPE
            || objective_identity.compressed_size != objective_payload.len() as u64
        {
            return Err(AtlasError::DigestMismatch);
        }
        objective_identity.verify_uncompressed(objective_payload)?;
        let objectives = AtlasObjectives::from_canonical_json(
            objective_payload,
            expected_map_id,
            &manifest.bsp.sha256,
            &atlas_sha256,
            artifact.origin,
            &artifact.l1,
        )?;
        if objective_identity.counts
            != std::collections::BTreeMap::from([(
                "objectives".to_owned(),
                objectives.objectives.len() as u64,
            )])
        {
            return Err(AtlasError::InvalidFormat(
                "objective artifact count differs from its manifest identity".to_owned(),
            ));
        }
        let objective_guide = ObjectiveGuide::build(&objectives, &artifact.l1)?;
        let hazard_components = HazardComponentField::build(&artifact.l1)?;
        let resident_bytes = artifact
            .resident_bytes_estimate()
            .checked_add(objectives.resident_bytes_estimate())
            .and_then(|value| value.checked_add(objective_guide.resident_bytes_estimate()))
            .and_then(|value| value.checked_add(hazard_components.resident_bytes_estimate()))
            .ok_or_else(|| {
                AtlasError::LimitExceeded("runtime Atlas resident estimate overflow".to_owned())
            })?;
        if resident_bytes > limits.max_atlas_resident_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "runtime Atlas/objective resident estimate {resident_bytes} > {}",
                limits.max_atlas_resident_bytes
            )));
        }
        Ok(Self {
            manifest,
            artifact: Arc::new(artifact),
            objectives: Arc::new(objectives),
            objective_guide: Arc::new(objective_guide),
            hazard_components: Arc::new(hazard_components),
            artifact_name: artifact_name.to_owned(),
            objective_artifact_name: objective_artifact_name.to_owned(),
            manifest_sha256: sha256_hex(manifest_bytes),
            atlas_sha256,
            objective_sha256: sha256_hex(objective_payload),
            map_epoch,
            counters: Arc::new(AtlasQueryCounters::default()),
        })
    }

    pub fn manifest(&self) -> &AtlasManifest {
        &self.manifest
    }

    pub fn artifact(&self) -> &AtlasArtifact {
        &self.artifact
    }

    pub fn artifact_name(&self) -> &str {
        &self.artifact_name
    }

    pub fn objectives(&self) -> &AtlasObjectives {
        &self.objectives
    }

    pub fn objective_artifact_name(&self) -> &str {
        &self.objective_artifact_name
    }

    pub fn atlas_sha256(&self) -> &str {
        &self.atlas_sha256
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn objective_sha256(&self) -> &str {
        &self.objective_sha256
    }

    pub fn map_epoch(&self) -> u64 {
        self.map_epoch
    }

    pub fn resident_bytes_estimate(&self) -> usize {
        self.artifact.resident_bytes_estimate()
            + self.objectives.resident_bytes_estimate()
            + self.objective_guide.resident_bytes_estimate()
            + self.hazard_components.resident_bytes_estimate()
    }

    pub fn shared_instance_count(&self) -> usize {
        Arc::strong_count(&self.artifact)
    }

    pub fn query_counters(&self) -> AtlasQueryCountersSnapshot {
        self.counters.snapshot()
    }

    pub fn recovery(
        &self,
        expected_map_epoch: u64,
        query: RecoveryQuery<'_>,
    ) -> AtlasResult<RecoveryFeatureBlock> {
        self.check_epoch(expected_map_epoch)?;
        let current = resolve_recovery_node(
            self.artifact.origin,
            &self.artifact.l1,
            query.world_position,
        )?;
        recovery_features_at(
            self.artifact.origin,
            &self.artifact.l1,
            &self.hazard_components,
            current,
            query,
        )
    }

    pub fn guide(
        &self,
        expected_map_epoch: u64,
        world_position: [f64; 3],
        yaw_degrees: f32,
        beliefs: &[ObjectiveBelief],
    ) -> AtlasResult<GuideFeatureBlock> {
        self.check_epoch(expected_map_epoch)?;
        let current =
            resolve_recovery_node(self.artifact.origin, &self.artifact.l1, world_position)?;
        self.guide_at(current, world_position, yaw_degrees, beliefs)
    }

    pub fn advisory_spatial_features(
        &self,
        expected_map_epoch: u64,
        recovery: RecoveryQuery<'_>,
        beliefs: &[ObjectiveBelief],
    ) -> AtlasResult<[f32; ADVISORY_SPATIAL_WIDTH]> {
        Ok(self
            .advisory_spatial_features_timed(expected_map_epoch, recovery, beliefs)?
            .values)
    }

    pub fn advisory_spatial_features_timed(
        &self,
        expected_map_epoch: u64,
        recovery: RecoveryQuery<'_>,
        beliefs: &[ObjectiveBelief],
    ) -> AtlasResult<TimedAdvisoryFeatures> {
        let total_started = Instant::now();
        self.check_epoch(expected_map_epoch)?;
        let atlas_started = Instant::now();
        let current = resolve_recovery_node(
            self.artifact.origin,
            &self.artifact.l1,
            recovery.world_position,
        )?;
        let atlas_lookup_ns = elapsed_ns(atlas_started);
        let recovery_started = Instant::now();
        let recovery_block = recovery_features_at(
            self.artifact.origin,
            &self.artifact.l1,
            &self.hazard_components,
            current,
            recovery,
        )?;
        let recovery_ns = elapsed_ns(recovery_started);
        let guide_started = Instant::now();
        let guide_block = self.guide_at(
            current,
            recovery.world_position,
            recovery.yaw_degrees,
            beliefs,
        )?;
        let guide_ns = elapsed_ns(guide_started);
        let mut values = [0.0; ADVISORY_SPATIAL_WIDTH];
        values[..RECOVERY_FEATURE_WIDTH].copy_from_slice(&recovery_block.values);
        values[RECOVERY_FEATURE_WIDTH..].copy_from_slice(&guide_block.values);
        let timing = AtlasQueryTiming {
            atlas_lookup_ns,
            recovery_ns,
            guide_ns,
            total_ns: elapsed_ns(total_started),
        };
        self.counters.record(timing);
        Ok(TimedAdvisoryFeatures { values, timing })
    }

    fn check_epoch(&self, expected_map_epoch: u64) -> AtlasResult<()> {
        if expected_map_epoch != self.map_epoch {
            return Err(AtlasError::InvalidFormat(format!(
                "runtime Atlas map epoch {} != query epoch {expected_map_epoch}",
                self.map_epoch
            )));
        }
        Ok(())
    }

    fn guide_at(
        &self,
        current: usize,
        world_position: [f64; 3],
        yaw_degrees: f32,
        beliefs: &[ObjectiveBelief],
    ) -> AtlasResult<GuideFeatureBlock> {
        self.objective_guide.features(
            &self.objectives,
            &self.artifact.l1,
            self.artifact.origin,
            current,
            world_position,
            yaw_degrees,
            beliefs,
        )
    }
}

/// One atomic map-epoch selector. Readers either retain the prior immutable
/// runtime or obtain the complete new runtime; absence and epoch mismatch fail
/// closed and never select a legacy implementation.
#[derive(Debug, Default)]
pub struct AtlasRuntimeSlot {
    current: RwLock<Option<Arc<AtlasRuntime>>>,
}

impl AtlasRuntimeSlot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn activate(&self, runtime: AtlasRuntime) -> AtlasResult<()> {
        let mut current = self
            .current
            .write()
            .map_err(|_| AtlasError::InvalidFormat("runtime slot lock is poisoned".to_owned()))?;
        if let Some(active) = current.as_ref() {
            if runtime.map_epoch < active.map_epoch {
                return Err(AtlasError::InvalidFormat(
                    "runtime slot refuses a map-epoch downgrade".to_owned(),
                ));
            }
            if runtime.map_epoch == active.map_epoch {
                if runtime.manifest_sha256 == active.manifest_sha256
                    && runtime.atlas_sha256 == active.atlas_sha256
                    && runtime.objective_sha256 == active.objective_sha256
                    && runtime.manifest.bsp.sha256 == active.manifest.bsp.sha256
                    && runtime.manifest.bsp.canonical_map_id == active.manifest.bsp.canonical_map_id
                {
                    return Ok(());
                }
                return Err(AtlasError::InvalidFormat(
                    "runtime slot refuses mixed identities in one map epoch".to_owned(),
                ));
            }
        }
        *current = Some(Arc::new(runtime));
        Ok(())
    }

    pub fn snapshot(&self, expected_map_epoch: u64) -> AtlasResult<Arc<AtlasRuntime>> {
        let current = self
            .current
            .read()
            .map_err(|_| AtlasError::InvalidFormat("runtime slot lock is poisoned".to_owned()))?;
        let runtime = current.as_ref().ok_or_else(|| {
            AtlasError::InvalidFormat("runtime slot has no admitted Atlas".to_owned())
        })?;
        runtime.check_epoch(expected_map_epoch)?;
        Ok(Arc::clone(runtime))
    }

    pub fn clear(&self, expected_map_epoch: u64) -> AtlasResult<()> {
        let mut current = self
            .current
            .write()
            .map_err(|_| AtlasError::InvalidFormat("runtime slot lock is poisoned".to_owned()))?;
        let runtime = current.as_ref().ok_or_else(|| {
            AtlasError::InvalidFormat("runtime slot has no admitted Atlas".to_owned())
        })?;
        runtime.check_epoch(expected_map_epoch)?;
        *current = None;
        Ok(())
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

pub fn advisory_spatial_feature_names() -> Vec<&'static str> {
    RECOVERY_FEATURE_NAMES
        .into_iter()
        .chain(GUIDE_FEATURE_NAMES)
        .collect()
}
