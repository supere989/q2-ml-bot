//! Per-client, persistent dynamic Lattice state.
//!
//! This module is deliberately separate from the map-static [`crate::atlas`]
//! representation. `DynState` owns only persistent L2 evidence and its
//! deterministic L3 mip. Live thermal tracks are an ephemeral input to feature
//! assembly and have no snapshot encoding path.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::io::{Cursor, Read};

use sha2::{Digest, Sha256};

use crate::atlas::{AtlasLevel, AtlasOrigin, GridIndex};

pub const DYN_MAGIC: &[u8; 8] = b"Q2LAT002";
pub const RETIRED_DYN_MAGIC: &[u8; 8] = b"Q2LAT001";
pub const DYN_SCHEMA_VERSION: u16 = 2;
pub const DYN_FEATURE_WIDTH: usize = 24;
pub const DYN_FEATURE_SCHEMA_SHA256: &str =
    "97d0c42da952717613a26fb8453c0b65ba8002e4db26eab48d85ffe81b020a22";
pub const DYN_CELL_ENCODED_BYTES: usize = 40;
pub const DYN_L2_CELL_SIZE: u32 = 64;
pub const DYN_L3_CELL_SIZE: u32 = 256;
pub const THERMAL_MAX_AGE_TICKS: u64 = 5;
pub const DYN_EVENT_SCHEMA: &str = "q2-dyn-named-event-v1";
pub const DYN_EVENT_NAMES: [&str; 5] =
    ["engagement", "threat", "opportunity", "self_fire", "death"];
pub const DYN_DECAY_INTERVAL_STEPS: u64 = 1024;
pub const DYN_RUNTIME_CHECKPOINT_MAGIC: &[u8; 8] = b"Q2DRT001";
pub const DYN_RUNTIME_CHECKPOINT_VERSION: u16 = 1;

const BYTE_ORDER_LITTLE: u16 = 0x454c;
const SNAPSHOT_HEADER_BYTES: usize = 208;
const RUNTIME_CHECKPOINT_HEADER_BYTES: usize = 208;
const COMPRESSION_ZSTD: u8 = 1;
const ZSTD_LEVEL: i8 = 3;
const COARSE_PARENT_BUDGET: usize = 4;
const RESIDENT_BYTES_PER_CELL: usize = 96;

pub type DynResult<T> = Result<T, DynError>;

#[derive(Debug)]
pub enum DynError {
    Coordinate(String),
    InvalidCell(String),
    InvalidFormat(String),
    LimitExceeded(String),
    MixedSchema { expected: u16, found: u16 },
    RetiredSchema,
    DigestMismatch,
    FenceMismatch(&'static str),
    StaleEnvironmentSteps { expected: u64, found: u64 },
    StaleClientLifeEpoch { expected: u64, found: u64 },
    StaleServerFrame { previous: u64, found: u64 },
    DuplicateEventId(u64),
    StaleEventId { previous: u64, found: u64 },
    Io(std::io::Error),
}

impl Display for DynError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coordinate(message) => write!(formatter, "invalid Dyn coordinate: {message}"),
            Self::InvalidCell(message) => write!(formatter, "invalid Dyn cell: {message}"),
            Self::InvalidFormat(message) => write!(formatter, "invalid Dyn format: {message}"),
            Self::LimitExceeded(message) => write!(formatter, "Dyn limit exceeded: {message}"),
            Self::MixedSchema { expected, found } => {
                write!(
                    formatter,
                    "mixed Dyn schema: expected {expected}, found {found}"
                )
            }
            Self::RetiredSchema => formatter.write_str("retired Q2LAT001 Dyn snapshot rejected"),
            Self::DigestMismatch => formatter.write_str("Dyn payload digest mismatch"),
            Self::FenceMismatch(field) => write!(formatter, "Dyn fence mismatch: {field}"),
            Self::StaleEnvironmentSteps { expected, found } => write!(
                formatter,
                "stale Dyn environment steps: expected {expected}, found {found}"
            ),
            Self::StaleClientLifeEpoch { expected, found } => write!(
                formatter,
                "stale Dyn client life epoch: expected {expected}, found {found}"
            ),
            Self::StaleServerFrame { previous, found } => write!(
                formatter,
                "stale Dyn server frame: previous {previous}, found {found}"
            ),
            Self::DuplicateEventId(event_id) => {
                write!(formatter, "duplicate Dyn event id {event_id}")
            }
            Self::StaleEventId { previous, found } => write!(
                formatter,
                "stale Dyn event id: previous {previous}, found {found}"
            ),
            Self::Io(error) => write!(formatter, "Dyn I/O error: {error}"),
        }
    }
}

impl std::error::Error for DynError {}

impl From<std::io::Error> for DynError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynLimits {
    pub max_clients: u32,
    pub max_l2_cells: usize,
    pub max_l3_cells: usize,
    pub max_materialized_cells: usize,
    pub max_resident_bytes: usize,
    pub max_uncompressed_snapshot_bytes: usize,
    pub max_compressed_snapshot_bytes: usize,
    pub batch_soft_compressed_bytes: usize,
    pub batch_hard_compressed_bytes: usize,
    pub batch_hard_resident_bytes: usize,
    pub max_events_per_transaction: usize,
    pub max_events_per_client_life: u64,
}

impl Default for DynLimits {
    fn default() -> Self {
        Self {
            max_clients: 4,
            max_l2_cells: 20_000,
            max_l3_cells: 20_000,
            max_materialized_cells: 20_000,
            // Four accepted clients must remain strictly below 8 MiB.
            max_resident_bytes: 2 * 1024 * 1024 - 1,
            max_uncompressed_snapshot_bytes: 2 * 1024 * 1024,
            max_compressed_snapshot_bytes: 2 * 1024 * 1024,
            batch_soft_compressed_bytes: 2 * 1024 * 1024,
            batch_hard_compressed_bytes: 8 * 1024 * 1024,
            batch_hard_resident_bytes: 8 * 1024 * 1024 - 1,
            max_events_per_transaction: DYN_EVENT_NAMES.len(),
            max_events_per_client_life: 10_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynFence {
    pub atlas_sha256: [u8; 32],
    pub map_sha256: [u8; 32],
    pub origin: AtlasOrigin,
    pub map_epoch: u64,
}

impl DynFence {
    pub fn validate(self) -> DynResult<()> {
        if self.atlas_sha256 == [0; 32] {
            return Err(DynError::InvalidFormat(
                "missing Atlas SHA-256 binding".to_owned(),
            ));
        }
        if self.map_sha256 == [0; 32] {
            return Err(DynError::InvalidFormat(
                "missing map SHA-256 identity".to_owned(),
            ));
        }
        if self.origin.0.iter().any(|value| value.rem_euclid(256) != 0) {
            return Err(DynError::InvalidFormat(
                "Dyn origin is not snapped to 256 units".to_owned(),
            ));
        }
        Ok(())
    }

    fn check(self, expected: Self) -> DynResult<()> {
        if self.atlas_sha256 != expected.atlas_sha256 {
            return Err(DynError::FenceMismatch("atlas_sha256"));
        }
        if self.map_sha256 != expected.map_sha256 {
            return Err(DynError::FenceMismatch("map_sha256"));
        }
        if self.origin != expected.origin {
            return Err(DynError::FenceMismatch("origin"));
        }
        if self.map_epoch != expected.map_epoch {
            return Err(DynError::FenceMismatch("map_epoch"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PersistentChannels {
    pub engagement: f32,
    pub threat: f32,
    pub opportunity: f32,
    pub self_fire: f32,
    pub deaths: f32,
}

impl PersistentChannels {
    pub const ZERO: Self = Self {
        engagement: 0.0,
        threat: 0.0,
        opportunity: 0.0,
        self_fire: 0.0,
        deaths: 0.0,
    };

    fn values(self) -> [f32; 5] {
        [
            self.engagement,
            self.threat,
            self.opportunity,
            self.self_fire,
            self.deaths,
        ]
    }

    fn from_values(values: [f32; 5]) -> Self {
        Self {
            engagement: values[0],
            threat: values[1],
            opportunity: values[2],
            self_fire: values[3],
            deaths: values[4],
        }
    }

    fn get(self, channel: PersistentChannel) -> f32 {
        self.values()[channel as usize]
    }

    fn add_assign(&mut self, other: Self) {
        self.engagement += other.engagement;
        self.threat += other.threat;
        self.opportunity += other.opportunity;
        self.self_fire += other.self_fire;
        self.deaths += other.deaths;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PersistentChannel {
    Engagement = 0,
    Threat = 1,
    Opportunity = 2,
    SelfFire = 3,
    Deaths = 4,
}

impl PersistentChannel {
    const ALL: [Self; 5] = [
        Self::Engagement,
        Self::Threat,
        Self::Opportunity,
        Self::SelfFire,
        Self::Deaths,
    ];
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynCell {
    pub channels: PersistentChannels,
    pub sample_mass: f32,
    pub confidence: f32,
}

impl DynCell {
    pub fn new(channels: PersistentChannels, sample_mass: f32, confidence: f32) -> DynResult<Self> {
        let mut result = Self {
            channels,
            sample_mass,
            confidence,
        };
        result.canonicalize_zero();
        result.validate()?;
        Ok(result)
    }

    fn canonicalize_zero(&mut self) {
        let mut values = self.channels.values();
        for value in &mut values {
            if *value == 0.0 {
                *value = 0.0;
            }
        }
        self.channels = PersistentChannels::from_values(values);
        if self.sample_mass == 0.0 {
            self.sample_mass = 0.0;
        }
        if self.confidence == 0.0 {
            self.confidence = 0.0;
        }
    }

    fn validate(self) -> DynResult<()> {
        for (name, value) in [
            ("engagement", self.channels.engagement),
            ("threat", self.channels.threat),
            ("opportunity", self.channels.opportunity),
            ("self_fire", self.channels.self_fire),
            ("deaths", self.channels.deaths),
            ("sample_mass", self.sample_mass),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(DynError::InvalidCell(format!(
                    "{name} must be finite and nonnegative"
                )));
            }
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(DynError::InvalidCell(
                "confidence must be finite and in 0..=1".to_owned(),
            ));
        }
        Ok(())
    }

    fn aggregate(&mut self, child: Self) {
        self.channels.add_assign(child.channels);
        self.sample_mass += child.sample_mass;
        self.confidence = self.confidence.max(child.confidence);
    }

    fn bitwise_eq(self, other: Self) -> bool {
        self.channels
            .values()
            .into_iter()
            .zip(other.channels.values())
            .all(|(left, right)| left.to_bits() == right.to_bits())
            && self.sample_mass.to_bits() == other.sample_mass.to_bits()
            && self.confidence.to_bits() == other.confidence.to_bits()
    }
}

#[derive(Clone, Debug)]
pub struct DynState {
    fence: DynFence,
    client_id: u32,
    client_count: u32,
    environment_steps: u64,
    l2: BTreeMap<GridIndex, DynCell>,
    l3: BTreeMap<GridIndex, DynCell>,
}

impl DynState {
    pub fn new(
        fence: DynFence,
        client_id: u32,
        client_count: u32,
        environment_steps: u64,
        limits: &DynLimits,
    ) -> DynResult<Self> {
        fence.validate()?;
        validate_client_identity(client_id, client_count, limits)?;
        Ok(Self {
            fence,
            client_id,
            client_count,
            environment_steps,
            l2: BTreeMap::new(),
            l3: BTreeMap::new(),
        })
    }

    pub fn fence(&self) -> DynFence {
        self.fence
    }

    pub fn client_id(&self) -> u32 {
        self.client_id
    }

    pub fn client_count(&self) -> u32 {
        self.client_count
    }

    pub fn environment_steps(&self) -> u64 {
        self.environment_steps
    }

    pub fn l2_len(&self) -> usize {
        self.l2.len()
    }

    pub fn l3_len(&self) -> usize {
        self.l3.len()
    }

    pub fn l2_cell(&self, index: GridIndex) -> Option<&DynCell> {
        self.l2.get(&index)
    }

    pub fn l3_cell(&self, index: GridIndex) -> Option<&DynCell> {
        self.l3.get(&index)
    }

    pub fn l2_cells(&self) -> impl ExactSizeIterator<Item = (&GridIndex, &DynCell)> {
        self.l2.iter()
    }

    pub fn l3_cells(&self) -> impl ExactSizeIterator<Item = (&GridIndex, &DynCell)> {
        self.l3.iter()
    }

    pub fn set_environment_steps(
        &mut self,
        expected_fence: DynFence,
        environment_steps: u64,
    ) -> DynResult<()> {
        self.fence.check(expected_fence)?;
        if environment_steps < self.environment_steps {
            return Err(DynError::StaleEnvironmentSteps {
                expected: self.environment_steps,
                found: environment_steps,
            });
        }
        self.environment_steps = environment_steps;
        Ok(())
    }

    fn advance_environment_steps(
        &mut self,
        expected_fence: DynFence,
        environment_steps: u64,
        limits: &DynLimits,
    ) -> DynResult<u64> {
        self.fence.check(expected_fence)?;
        if environment_steps < self.environment_steps {
            return Err(DynError::StaleEnvironmentSteps {
                expected: self.environment_steps,
                found: environment_steps,
            });
        }
        let previous_interval = self.environment_steps / DYN_DECAY_INTERVAL_STEPS;
        let next_interval = environment_steps / DYN_DECAY_INTERVAL_STEPS;
        let elapsed_intervals = next_interval - previous_interval;
        if elapsed_intervals == 0 {
            self.environment_steps = environment_steps;
            return Ok(0);
        }

        // Power-of-two decay is deliberate: applying N intervals at once or
        // through any transaction partition produces the same IEEE-754 bits.
        // Evidence that is older than the f32 normal exponent range is exactly
        // zero and its sparse cell is removed.
        let scale = if elapsed_intervals >= 126 {
            0.0
        } else {
            2.0_f32.powi(-(elapsed_intervals as i32))
        };
        self.l2.retain(|_, cell| {
            let mut values = cell.channels.values();
            for value in &mut values {
                *value *= scale;
            }
            cell.channels = PersistentChannels::from_values(values);
            cell.sample_mass *= scale;
            cell.confidence *= scale;
            cell.canonicalize_zero();
            cell.channels.values().into_iter().any(|value| value != 0.0) || cell.sample_mass != 0.0
        });
        self.l3 = derive_l3(&self.l2)?;
        self.environment_steps = environment_steps;
        self.validate(limits)?;
        Ok(elapsed_intervals)
    }

    pub fn upsert_l2(
        &mut self,
        expected_fence: DynFence,
        index: GridIndex,
        mut cell: DynCell,
        limits: &DynLimits,
    ) -> DynResult<Option<DynCell>> {
        self.fence.check(expected_fence)?;
        cell.canonicalize_zero();
        cell.validate()?;
        let parent = index.parent();
        let previous_parent = self.l3.get(&parent).copied();
        let previous = self.l2.insert(index, cell);
        let result = self
            .rebuild_parent(parent)
            .and_then(|()| self.validate_budgets(limits));
        if let Err(error) = result {
            if let Some(old) = previous {
                self.l2.insert(index, old);
            } else {
                self.l2.remove(&index);
            }
            if let Some(old_parent) = previous_parent {
                self.l3.insert(parent, old_parent);
            } else {
                self.l3.remove(&parent);
            }
            return Err(error);
        }
        Ok(previous)
    }

    pub fn remove_l2(
        &mut self,
        expected_fence: DynFence,
        index: GridIndex,
        limits: &DynLimits,
    ) -> DynResult<Option<DynCell>> {
        self.fence.check(expected_fence)?;
        let previous = self.l2.remove(&index);
        if previous.is_none() {
            return Ok(None);
        }
        self.rebuild_parent(index.parent())?;
        if let Err(error) = self.validate_budgets(limits) {
            self.l2.insert(index, previous.expect("checked above"));
            self.rebuild_parent(index.parent())?;
            return Err(error);
        }
        Ok(previous)
    }

    pub fn resident_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>() + (self.l2.len() + self.l3.len()) * RESIDENT_BYTES_PER_CELL
    }

    pub fn validate(&self, limits: &DynLimits) -> DynResult<()> {
        self.fence.validate()?;
        validate_client_identity(self.client_id, self.client_count, limits)?;
        self.validate_budgets(limits)?;
        for cell in self.l2.values().chain(self.l3.values()) {
            cell.validate()?;
        }
        let expected_l3 = derive_l3(&self.l2)?;
        if expected_l3.len() != self.l3.len()
            || expected_l3.iter().any(|(index, expected)| {
                self.l3
                    .get(index)
                    .is_none_or(|actual| !actual.bitwise_eq(*expected))
            })
        {
            return Err(DynError::InvalidFormat(
                "Dyn L3 is not the canonical derived mip of L2".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_budgets(&self, limits: &DynLimits) -> DynResult<()> {
        if self.l2.len() > limits.max_l2_cells {
            return Err(DynError::LimitExceeded(format!(
                "L2 cells {} > {}",
                self.l2.len(),
                limits.max_l2_cells
            )));
        }
        if self.l3.len() > limits.max_l3_cells {
            return Err(DynError::LimitExceeded(format!(
                "L3 cells {} > {}",
                self.l3.len(),
                limits.max_l3_cells
            )));
        }
        let materialized = self.l2.len().checked_add(self.l3.len()).ok_or_else(|| {
            DynError::LimitExceeded("materialized cell count overflow".to_owned())
        })?;
        if materialized > limits.max_materialized_cells {
            return Err(DynError::LimitExceeded(format!(
                "materialized cells {materialized} > {}",
                limits.max_materialized_cells
            )));
        }
        let resident = self.resident_bytes_estimate();
        if resident > limits.max_resident_bytes {
            return Err(DynError::LimitExceeded(format!(
                "resident estimate {resident} > {}",
                limits.max_resident_bytes
            )));
        }
        Ok(())
    }

    pub fn feature_block(&self, input: DynFeatureInput) -> DynResult<DynFeatureBlock> {
        self.fence.check(input.fence)?;
        validate_feature_input(input)?;
        let current_index = self
            .fence
            .origin
            .index(input.world_position, AtlasLevel::L2)
            .map_err(|error| DynError::Coordinate(error.to_string()))?;
        let mut values = [0.0; DYN_FEATURE_WIDTH];
        if let Some(current) = self.l2.get(&current_index) {
            for (index, score) in current.channels.values()[..4].iter().enumerate() {
                values[index] = normalize_score(*score, input.score_scale);
            }
            values[DynFeatureIndex::CurrentConfidence as usize] = current.confidence;
        }

        let nearest = self.nearest_signals(input)?;
        write_signal(
            &mut values,
            DynFeatureIndex::ImmediateThermalForward as usize,
            if let Some(thermal) = input.thermal {
                signal_from_world_point(
                    input.world_position,
                    thermal.world_point,
                    thermal.heat.clamp(0.0, 1.0),
                    input.yaw_degrees,
                    input.search_radius,
                )
            } else {
                nearest[PersistentChannel::Engagement as usize]
            },
        );
        write_signal(
            &mut values,
            DynFeatureIndex::CombatThreatForward as usize,
            nearest[PersistentChannel::Threat as usize],
        );
        write_signal(
            &mut values,
            DynFeatureIndex::OpportunityForward as usize,
            nearest[PersistentChannel::Opportunity as usize],
        );
        write_signal(
            &mut values,
            DynFeatureIndex::SelfFireForward as usize,
            nearest[PersistentChannel::SelfFire as usize],
        );
        values[DynFeatureIndex::WinMargin as usize] = input.survivability[0].clamp(-1.0, 1.0);
        values[DynFeatureIndex::EffectiveHealthNorm as usize] =
            input.survivability[1].clamp(0.0, 1.0);
        values[DynFeatureIndex::OwnDpsShare as usize] = input.survivability[2].clamp(0.0, 1.0);

        Ok(DynFeatureBlock {
            values,
            nearest_death_score: nearest[PersistentChannel::Deaths as usize].score,
        })
    }

    fn rebuild_parent(&mut self, parent: GridIndex) -> DynResult<()> {
        let aggregate = aggregate_parent(&self.l2, parent)?;
        if let Some(cell) = aggregate {
            self.l3.insert(parent, cell);
        } else {
            self.l3.remove(&parent);
        }
        Ok(())
    }

    fn nearest_signals(&self, input: DynFeatureInput) -> DynResult<[Signal; 5]> {
        let center = self
            .fence
            .origin
            .index(input.world_position, AtlasLevel::L3)
            .map_err(|error| DynError::Coordinate(error.to_string()))?;
        let coarse_radius =
            (input.search_radius / AtlasLevel::L3.cell_size() as f32).ceil() as i64 + 1;
        let x_min = i64::from(center.x)
            .saturating_sub(coarse_radius)
            .max(i64::from(i32::MIN));
        let x_max = i64::from(center.x)
            .saturating_add(coarse_radius)
            .min(i64::from(i32::MAX));
        let y_min = i64::from(center.y)
            .saturating_sub(coarse_radius)
            .max(i64::from(i32::MIN));
        let y_max = i64::from(center.y)
            .saturating_add(coarse_radius)
            .min(i64::from(i32::MAX));
        let z_min = i64::from(center.z)
            .saturating_sub(coarse_radius)
            .max(i64::from(i32::MIN));
        let z_max = i64::from(center.z)
            .saturating_add(coarse_radius)
            .min(i64::from(i32::MAX));
        let mut parents: [Vec<CoarseCandidate>; 5] = std::array::from_fn(|_| Vec::new());

        for z in z_min..=z_max {
            for y in y_min..=y_max {
                let lower = GridIndex::new(x_min as i32, y as i32, z as i32);
                let upper = GridIndex::new(x_max as i32, y as i32, z as i32);
                for (index, cell) in self.l3.range(lower..=upper) {
                    let world_center = self.fence.origin.center(*index, AtlasLevel::L3);
                    let distance = distance(input.world_position, world_center);
                    if distance
                        > f64::from(input.search_radius)
                            + (3.0_f64).sqrt() * AtlasLevel::L3.cell_size() as f64 / 2.0
                    {
                        continue;
                    }
                    for channel in PersistentChannel::ALL {
                        let raw = cell.channels.get(channel);
                        if raw <= 0.0 {
                            continue;
                        }
                        let score = normalize_score(raw, input.score_scale) * cell.confidence;
                        let rank = score
                            / ((distance as f32 / AtlasLevel::L3.cell_size() as f32).max(1.0));
                        push_coarse_candidate(
                            &mut parents[channel as usize],
                            CoarseCandidate {
                                index: *index,
                                rank,
                            },
                        );
                    }
                }
            }
        }

        let mut result = [Signal::default(); 5];
        for channel in PersistentChannel::ALL {
            let mut best: Option<FineCandidate> = None;
            for parent in &parents[channel as usize] {
                let minimum = parent
                    .index
                    .child_min()
                    .map_err(|error| DynError::Coordinate(error.to_string()))?;
                for z in 0..4 {
                    for y in 0..4 {
                        for x in 0..4 {
                            let index = GridIndex::new(minimum.x + x, minimum.y + y, minimum.z + z);
                            let Some(cell) = self.l2.get(&index) else {
                                continue;
                            };
                            let raw = cell.channels.get(channel);
                            if raw <= 0.0 {
                                continue;
                            }
                            let world_center = self.fence.origin.center(index, AtlasLevel::L2);
                            let distance = distance(input.world_position, world_center);
                            if distance > f64::from(input.search_radius) {
                                continue;
                            }
                            let score = normalize_score(raw, input.score_scale) * cell.confidence;
                            let candidate = FineCandidate {
                                index,
                                world_center,
                                score,
                                rank: score
                                    / ((distance as f32 / AtlasLevel::L2.cell_size() as f32)
                                        .max(1.0)),
                            };
                            if best.is_none_or(|current| candidate.better_than(current)) {
                                best = Some(candidate);
                            }
                        }
                    }
                }
            }
            if let Some(candidate) = best {
                result[channel as usize] = signal_from_world_point(
                    input.world_position,
                    candidate.world_center,
                    candidate.score,
                    input.yaw_degrees,
                    input.search_radius,
                );
            }
        }
        Ok(result)
    }
}

fn validate_client_identity(
    client_id: u32,
    client_count: u32,
    limits: &DynLimits,
) -> DynResult<()> {
    if client_count == 0 || client_count > limits.max_clients {
        return Err(DynError::LimitExceeded(format!(
            "client count {client_count} is outside 1..={}",
            limits.max_clients
        )));
    }
    if client_id >= client_count {
        return Err(DynError::InvalidFormat(format!(
            "client id {client_id} is not below client count {client_count}"
        )));
    }
    Ok(())
}

fn aggregate_parent(
    l2: &BTreeMap<GridIndex, DynCell>,
    parent: GridIndex,
) -> DynResult<Option<DynCell>> {
    let minimum = parent
        .child_min()
        .map_err(|error| DynError::Coordinate(error.to_string()))?;
    let mut aggregate = DynCell::default();
    let mut found = false;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let child = GridIndex::new(minimum.x + x, minimum.y + y, minimum.z + z);
                if let Some(cell) = l2.get(&child) {
                    aggregate.aggregate(*cell);
                    found = true;
                }
            }
        }
    }
    if found {
        aggregate.validate()?;
        Ok(Some(aggregate))
    } else {
        Ok(None)
    }
}

fn derive_l3(l2: &BTreeMap<GridIndex, DynCell>) -> DynResult<BTreeMap<GridIndex, DynCell>> {
    let parents: BTreeSet<_> = l2.keys().map(|index| index.parent()).collect();
    let mut l3 = BTreeMap::new();
    for parent in parents {
        if let Some(cell) = aggregate_parent(l2, parent)? {
            l3.insert(parent, cell);
        }
    }
    Ok(l3)
}

#[derive(Clone, Copy, Debug)]
pub struct ThermalSignal {
    pub target_id: u64,
    pub world_point: [f64; 3],
    pub heat: f32,
    pub observed_tick: u64,
}

#[derive(Clone, Copy, Debug)]
struct ThermalTrack {
    signal: ThermalSignal,
    expires_tick: u64,
}

/// Ephemeral per-client target heat. This type intentionally has no serde or
/// snapshot implementation and is not owned by [`DynState`].
#[derive(Clone, Debug, Default)]
pub struct ThermalOverlay {
    tracks: BTreeMap<u64, ThermalTrack>,
}

impl ThermalOverlay {
    pub fn observe(
        &mut self,
        target_id: u64,
        world_point: [f64; 3],
        heat: f32,
        tick: u64,
        age_ticks: u64,
    ) -> DynResult<()> {
        if age_ticks == 0 || age_ticks > THERMAL_MAX_AGE_TICKS {
            return Err(DynError::LimitExceeded(format!(
                "thermal age {age_ticks} is outside 1..={THERMAL_MAX_AGE_TICKS}"
            )));
        }
        if world_point.iter().any(|value| !value.is_finite()) {
            return Err(DynError::InvalidCell(
                "thermal world point must be finite".to_owned(),
            ));
        }
        if !heat.is_finite() || !(0.0..=1.0).contains(&heat) {
            return Err(DynError::InvalidCell(
                "thermal heat must be finite and in 0..=1".to_owned(),
            ));
        }
        let expires_tick = tick
            .checked_add(age_ticks)
            .ok_or_else(|| DynError::LimitExceeded("thermal expiry overflow".to_owned()))?;
        self.tracks.insert(
            target_id,
            ThermalTrack {
                signal: ThermalSignal {
                    target_id,
                    world_point,
                    heat,
                    observed_tick: tick,
                },
                expires_tick,
            },
        );
        Ok(())
    }

    pub fn expire(&mut self, tick: u64) {
        self.tracks.retain(|_, track| tick <= track.expires_tick);
    }

    pub fn strongest(&self, tick: u64) -> Option<ThermalSignal> {
        self.tracks
            .values()
            .filter(|track| tick <= track.expires_tick)
            .map(|track| track.signal)
            .max_by(|left, right| {
                left.heat
                    .total_cmp(&right.heat)
                    .then_with(|| right.target_id.cmp(&left.target_id))
            })
    }

    pub fn clear(&mut self) {
        self.tracks.clear();
    }

    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynFeatureInput {
    pub fence: DynFence,
    pub world_position: [f64; 3],
    pub yaw_degrees: f32,
    pub thermal: Option<ThermalSignal>,
    pub survivability: [f32; 3],
    pub search_radius: f32,
    pub score_scale: f32,
}

fn validate_feature_input(input: DynFeatureInput) -> DynResult<()> {
    if input.world_position.iter().any(|value| !value.is_finite())
        || !input.yaw_degrees.is_finite()
        || input.survivability.iter().any(|value| !value.is_finite())
    {
        return Err(DynError::InvalidCell(
            "feature input contains a non-finite value".to_owned(),
        ));
    }
    if !input.search_radius.is_finite() || input.search_radius < AtlasLevel::L2.cell_size() as f32 {
        return Err(DynError::InvalidCell(
            "feature search radius must be finite and at least 64".to_owned(),
        ));
    }
    if !input.score_scale.is_finite() || input.score_scale <= 0.0 {
        return Err(DynError::InvalidCell(
            "feature score scale must be finite and positive".to_owned(),
        ));
    }
    if let Some(thermal) = input.thermal
        && (thermal.world_point.iter().any(|value| !value.is_finite())
            || !thermal.heat.is_finite()
            || !(0.0..=1.0).contains(&thermal.heat))
    {
        return Err(DynError::InvalidCell(
            "thermal feature input is invalid".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum DynFeatureIndex {
    CurrentEngagement = 0,
    CurrentThreat = 1,
    CurrentOpportunity = 2,
    CurrentSelfFire = 3,
    CurrentConfidence = 4,
    ImmediateThermalForward = 5,
    ImmediateThermalQuakeRight = 6,
    ImmediateThermalUp = 7,
    ImmediateThermalHeat = 8,
    CombatThreatForward = 9,
    CombatThreatQuakeRight = 10,
    CombatThreatUp = 11,
    CombatThreatScore = 12,
    OpportunityForward = 13,
    OpportunityQuakeRight = 14,
    OpportunityUp = 15,
    OpportunityScore = 16,
    SelfFireForward = 17,
    SelfFireQuakeRight = 18,
    SelfFireUp = 19,
    SelfFireScore = 20,
    WinMargin = 21,
    EffectiveHealthNorm = 22,
    OwnDpsShare = 23,
}

pub const DYN_FEATURE_NAMES: [&str; DYN_FEATURE_WIDTH] = [
    "dyn_l2_current_engagement",
    "dyn_l2_current_threat",
    "dyn_l2_current_opportunity",
    "dyn_l2_current_self_fire",
    "dyn_l2_current_confidence",
    "dyn_immediate_thermal_forward",
    "dyn_immediate_thermal_quake_right",
    "dyn_immediate_thermal_up",
    "dyn_immediate_thermal_heat",
    "dyn_combat_threat_forward",
    "dyn_combat_threat_quake_right",
    "dyn_combat_threat_up",
    "dyn_combat_threat_score",
    "dyn_opportunity_forward",
    "dyn_opportunity_quake_right",
    "dyn_opportunity_up",
    "dyn_opportunity_score",
    "dyn_self_fire_forward",
    "dyn_self_fire_quake_right",
    "dyn_self_fire_up",
    "dyn_self_fire_score",
    "dyn_win_margin",
    "dyn_effective_health_norm",
    "dyn_own_dps_share",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynFeatureBlock {
    pub values: [f32; DYN_FEATURE_WIDTH],
    pub nearest_death_score: f32,
}

/// The only persistent live Dyn deposit vocabulary.
///
/// Each accepted event deposits one unit into exactly one named channel and
/// one unit of sample mass at its authoritative world point. Callers cannot
/// submit scores, confidence, packed cells, or precomputed Dyn24 values.
/// Production mapping is intentionally public and factual: engagement is
/// positive observed damage dealt at a current or at-most-five-tick remembered
/// visible target point; threat is positive observed damage taken at the own
/// point; opportunity is a newly actionable or new-L2 visible damageable
/// target point; self-fire is an accepted public action-echo shot edge; and
/// death is the public death reward/health terminal at the own point.
/// Unchanged presence/held input never repeats a deposit. Private causal or
/// reward-only telemetry is never a deposit source.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(usize)]
pub enum DynEventKind {
    Engagement,
    Threat,
    Opportunity,
    SelfFire,
    Death,
}

impl DynEventKind {
    pub fn name(self) -> &'static str {
        DYN_EVENT_NAMES[self as usize]
    }

    pub fn code(self) -> u64 {
        self as u64 + 1
    }

    pub fn parse(name: &str) -> DynResult<Self> {
        match name {
            "engagement" => Ok(Self::Engagement),
            "threat" => Ok(Self::Threat),
            "opportunity" => Ok(Self::Opportunity),
            "self_fire" => Ok(Self::SelfFire),
            "death" => Ok(Self::Death),
            _ => Err(DynError::InvalidFormat(format!(
                "unknown Dyn event kind {name:?}"
            ))),
        }
    }

    fn channel(self) -> PersistentChannel {
        match self {
            Self::Engagement => PersistentChannel::Engagement,
            Self::Threat => PersistentChannel::Threat,
            Self::Opportunity => PersistentChannel::Opportunity,
            Self::SelfFire => PersistentChannel::SelfFire,
            Self::Death => PersistentChannel::Deaths,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynNamedEvent {
    /// Exact `(server_frame << 3) | kind_code` identity, with kind codes 1..5.
    pub event_id: u64,
    pub kind: DynEventKind,
    pub world_position: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct DynIngestBatch<'a> {
    pub fence: DynFence,
    pub client_id: u32,
    pub client_life_epoch: u64,
    pub server_frame: u64,
    /// Compare-and-swap predecessor; prevents two producers from advancing
    /// the same runtime from different environment-step lineages.
    pub expected_environment_steps: u64,
    pub environment_steps: u64,
    pub events: &'a [DynNamedEvent],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynIngestReport {
    pub events_applied: usize,
    pub cells_updated: usize,
    pub decay_intervals: u64,
    pub environment_steps: u64,
    pub client_life_epoch: u64,
    pub server_frame: u64,
    pub last_event_id: u64,
}

/// Query accepted by the sole live Q2LAT002 feature adapter.
///
/// The adapter binds Atlas/map/origin and client identity at construction. A
/// query must present the exact live life/frame/environment cursor established
/// by the preceding transaction, so cached or pre-ingest calls fail closed.
#[derive(Clone, Copy, Debug)]
pub struct DynRuntimeQuery {
    pub map_epoch: u64,
    pub client_id: u32,
    pub client_life_epoch: u64,
    pub environment_steps: u64,
    pub server_frame: u64,
    pub world_position: [f64; 3],
    pub yaw_degrees: f32,
    pub thermal: Option<ThermalSignal>,
    pub survivability: [f32; 3],
    pub search_radius: f32,
    pub score_scale: f32,
}

/// Fail-closed per-client live Q2LAT002 event lattice and Dyn24 adapter.
#[derive(Clone, Debug)]
pub struct DynRuntime {
    state: DynState,
    snapshot_sha256: String,
    client_life_epoch: u64,
    server_frame: u64,
    last_event_id: u64,
    life_event_count: u64,
    accepted_event_count: u64,
    limits: DynLimits,
}

impl DynRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn from_snapshot(
        snapshot: &[u8],
        expected_fence: DynFence,
        expected_client_id: u32,
        expected_client_count: u32,
        expected_environment_steps: u64,
        limits: &DynLimits,
    ) -> DynResult<Self> {
        let state = decode_snapshot(snapshot, expected_fence, limits)?;
        if state.client_id() != expected_client_id {
            return Err(DynError::FenceMismatch("client_id"));
        }
        if state.client_count() != expected_client_count {
            return Err(DynError::FenceMismatch("client_count"));
        }
        if state.environment_steps() != expected_environment_steps {
            return Err(DynError::StaleEnvironmentSteps {
                expected: expected_environment_steps,
                found: state.environment_steps(),
            });
        }
        if encode_snapshot(&state, limits)? != snapshot {
            return Err(DynError::InvalidFormat(
                "Q2LAT002 snapshot is not the canonical byte encoding".to_owned(),
            ));
        }
        Ok(Self {
            state,
            snapshot_sha256: format!("{:x}", Sha256::digest(snapshot)),
            client_life_epoch: 0,
            server_frame: 0,
            last_event_id: 0,
            life_event_count: 0,
            accepted_event_count: 0,
            limits: limits.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint(
        checkpoint: &[u8],
        expected_checkpoint_sha256: [u8; 32],
        expected_fence: DynFence,
        expected_client_id: u32,
        expected_client_count: u32,
        expected_environment_steps: u64,
        expected_client_life_epoch: u64,
        expected_server_frame: u64,
        limits: &DynLimits,
    ) -> DynResult<Self> {
        expected_fence.validate()?;
        if <[u8; 32]>::from(Sha256::digest(checkpoint)) != expected_checkpoint_sha256 {
            return Err(DynError::DigestMismatch);
        }
        let header = RuntimeCheckpointHeader::decode(checkpoint, limits)?;
        header.fence.check(expected_fence)?;
        if header.client_id != expected_client_id {
            return Err(DynError::FenceMismatch("client_id"));
        }
        if header.client_count != expected_client_count {
            return Err(DynError::FenceMismatch("client_count"));
        }
        if header.environment_steps != expected_environment_steps {
            return Err(DynError::StaleEnvironmentSteps {
                expected: expected_environment_steps,
                found: header.environment_steps,
            });
        }
        if header.client_life_epoch != expected_client_life_epoch {
            return Err(DynError::StaleClientLifeEpoch {
                expected: expected_client_life_epoch,
                found: header.client_life_epoch,
            });
        }
        if header.server_frame != expected_server_frame {
            return Err(DynError::StaleServerFrame {
                previous: expected_server_frame,
                found: header.server_frame,
            });
        }
        let snapshot = checkpoint
            .get(RUNTIME_CHECKPOINT_HEADER_BYTES..)
            .ok_or_else(|| DynError::InvalidFormat("truncated runtime checkpoint".to_owned()))?;
        if snapshot.len() != header.snapshot_len
            || <[u8; 32]>::from(Sha256::digest(snapshot)) != header.snapshot_sha256
        {
            return Err(DynError::DigestMismatch);
        }
        let state = decode_snapshot(snapshot, expected_fence, limits)?;
        if encode_snapshot(&state, limits)? != snapshot {
            return Err(DynError::InvalidFormat(
                "checkpoint embeds a noncanonical Q2LAT002 snapshot".to_owned(),
            ));
        }
        if state.client_id() != header.client_id
            || state.client_count() != header.client_count
            || state.environment_steps() != header.environment_steps
        {
            return Err(DynError::InvalidFormat(
                "runtime checkpoint and embedded Q2LAT002 identity differ".to_owned(),
            ));
        }
        validate_runtime_cursor(
            header.client_life_epoch,
            header.server_frame,
            header.last_event_id,
            header.life_event_count,
            header.accepted_event_count,
            limits,
        )?;
        Ok(Self {
            state,
            snapshot_sha256: header
                .snapshot_sha256
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            client_life_epoch: header.client_life_epoch,
            server_frame: header.server_frame,
            last_event_id: header.last_event_id,
            life_event_count: header.life_event_count,
            accepted_event_count: header.accepted_event_count,
            limits: limits.clone(),
        })
    }

    pub fn state(&self) -> &DynState {
        &self.state
    }

    pub fn snapshot_sha256(&self) -> &str {
        &self.snapshot_sha256
    }

    pub fn client_life_epoch(&self) -> u64 {
        self.client_life_epoch
    }

    pub fn server_frame(&self) -> u64 {
        self.server_frame
    }

    pub fn last_event_id(&self) -> u64 {
        self.last_event_id
    }

    pub fn accepted_event_count(&self) -> u64 {
        self.accepted_event_count
    }

    pub fn snapshot_bytes(&self) -> DynResult<Vec<u8>> {
        encode_snapshot(&self.state, &self.limits)
    }

    pub fn checkpoint_bytes(&self) -> DynResult<Vec<u8>> {
        let snapshot = self.snapshot_bytes()?;
        let snapshot_sha256: [u8; 32] = Sha256::digest(&snapshot).into();
        let total = RUNTIME_CHECKPOINT_HEADER_BYTES
            .checked_add(snapshot.len())
            .ok_or_else(|| DynError::LimitExceeded("checkpoint size overflow".to_owned()))?;
        let maximum = RUNTIME_CHECKPOINT_HEADER_BYTES
            .checked_add(self.limits.max_compressed_snapshot_bytes)
            .ok_or_else(|| DynError::LimitExceeded("checkpoint limit overflow".to_owned()))?;
        if total > maximum {
            return Err(DynError::LimitExceeded(format!(
                "runtime checkpoint bytes {total} > {maximum}"
            )));
        }
        let mut output = Vec::with_capacity(total);
        output.extend_from_slice(DYN_RUNTIME_CHECKPOINT_MAGIC);
        push_u16(&mut output, DYN_RUNTIME_CHECKPOINT_VERSION);
        push_u16(&mut output, 0);
        push_u32(&mut output, RUNTIME_CHECKPOINT_HEADER_BYTES as u32);
        push_u64(&mut output, snapshot.len() as u64);
        output.extend_from_slice(&snapshot_sha256);
        output.extend_from_slice(&self.state.fence().atlas_sha256);
        output.extend_from_slice(&self.state.fence().map_sha256);
        for value in self.state.fence().origin.0 {
            push_i64(&mut output, value);
        }
        push_u64(&mut output, self.state.fence().map_epoch);
        push_u32(&mut output, self.state.client_id());
        push_u32(&mut output, self.state.client_count());
        push_u64(&mut output, self.state.environment_steps());
        push_u64(&mut output, self.client_life_epoch);
        push_u64(&mut output, self.server_frame);
        push_u64(&mut output, self.last_event_id);
        push_u64(&mut output, self.life_event_count);
        push_u64(&mut output, self.accepted_event_count);
        debug_assert_eq!(output.len(), RUNTIME_CHECKPOINT_HEADER_BYTES);
        output.extend_from_slice(&snapshot);
        Ok(output)
    }

    pub fn checkpoint_sha256(&self) -> DynResult<String> {
        Ok(format!("{:x}", Sha256::digest(self.checkpoint_bytes()?)))
    }

    /// Low-level qualification/checkpoint primitive.
    ///
    /// Production frame assembly must use [`Self::commit_frame`]. Calling
    /// `ingest_events` and [`Self::feature_block`] separately is not atomic.
    pub fn ingest_events(&mut self, batch: DynIngestBatch<'_>) -> DynResult<DynIngestReport> {
        self.state.fence().check(batch.fence)?;
        if batch.client_id != self.state.client_id() {
            return Err(DynError::FenceMismatch("client_id"));
        }
        if batch.client_life_epoch == 0 {
            return Err(DynError::InvalidFormat(
                "client life epoch must be nonzero".to_owned(),
            ));
        }
        let advances_life = if self.client_life_epoch == 0 {
            true
        } else if batch.client_life_epoch < self.client_life_epoch {
            return Err(DynError::StaleClientLifeEpoch {
                expected: self.client_life_epoch,
                found: batch.client_life_epoch,
            });
        } else if batch.client_life_epoch == self.client_life_epoch {
            false
        } else if self.client_life_epoch.checked_add(1) == Some(batch.client_life_epoch) {
            true
        } else {
            return Err(DynError::InvalidFormat(
                "client life epoch skipped an admitted life".to_owned(),
            ));
        };
        if batch.server_frame == 0 || batch.server_frame <= self.server_frame {
            return Err(DynError::StaleServerFrame {
                previous: self.server_frame,
                found: batch.server_frame,
            });
        }
        if batch.expected_environment_steps != self.state.environment_steps() {
            return Err(DynError::StaleEnvironmentSteps {
                expected: self.state.environment_steps(),
                found: batch.expected_environment_steps,
            });
        }
        if batch.events.len() > self.limits.max_events_per_transaction {
            return Err(DynError::LimitExceeded(format!(
                "events per transaction {} > {}",
                batch.events.len(),
                self.limits.max_events_per_transaction
            )));
        }

        let baseline_event_id = if advances_life { 0 } else { self.last_event_id };
        let baseline_life_events = if advances_life {
            0
        } else {
            self.life_event_count
        };
        let next_life_events = baseline_life_events
            .checked_add(batch.events.len() as u64)
            .ok_or_else(|| DynError::LimitExceeded("life event count overflow".to_owned()))?;
        if next_life_events > self.limits.max_events_per_client_life {
            return Err(DynError::LimitExceeded(format!(
                "events per client life {next_life_events} > {}",
                self.limits.max_events_per_client_life
            )));
        }
        let next_accepted_events = self
            .accepted_event_count
            .checked_add(batch.events.len() as u64)
            .ok_or_else(|| DynError::LimitExceeded("accepted event count overflow".to_owned()))?;

        let mut ordered = batch.events.to_vec();
        ordered.sort_by_key(|event| event.event_id);
        let event_frame_prefix = batch
            .server_frame
            .checked_mul(8)
            .ok_or_else(|| DynError::LimitExceeded("event ID frame prefix overflow".to_owned()))?;
        let mut previous_event_id = None;
        for event in &ordered {
            if event.event_id == 0 {
                return Err(DynError::InvalidFormat(
                    "Dyn event id must be nonzero".to_owned(),
                ));
            }
            if previous_event_id == Some(event.event_id) {
                return Err(DynError::DuplicateEventId(event.event_id));
            }
            let expected_event_id = event_frame_prefix | event.kind.code();
            if event.event_id != expected_event_id {
                return Err(DynError::InvalidFormat(format!(
                    "Dyn event id {} differs from frame/kind identity {expected_event_id}",
                    event.event_id
                )));
            }
            if event.event_id <= baseline_event_id {
                return Err(DynError::StaleEventId {
                    previous: baseline_event_id,
                    found: event.event_id,
                });
            }
            if event.world_position.iter().any(|value| !value.is_finite()) {
                return Err(DynError::InvalidCell(
                    "Dyn event world point must be finite".to_owned(),
                ));
            }
            previous_event_id = Some(event.event_id);
        }

        let mut next = self.state.clone();
        let decay_intervals =
            next.advance_environment_steps(batch.fence, batch.environment_steps, &self.limits)?;
        let mut deposits: BTreeMap<GridIndex, [u32; 5]> = BTreeMap::new();
        for event in &ordered {
            let index = batch
                .fence
                .origin
                .index(event.world_position, AtlasLevel::L2)
                .map_err(|error| DynError::Coordinate(error.to_string()))?;
            let counts = deposits.entry(index).or_default();
            counts[event.kind.channel() as usize] = counts[event.kind.channel() as usize]
                .checked_add(1)
                .ok_or_else(|| DynError::LimitExceeded("cell event count overflow".to_owned()))?;
        }
        for (index, counts) in &deposits {
            let mut cell = next.l2.get(index).copied().unwrap_or_default();
            let mut values = cell.channels.values();
            let mut sample_delta = 0_u32;
            for (value, count) in values.iter_mut().zip(*counts) {
                *value += count as f32;
                sample_delta = sample_delta.checked_add(count).ok_or_else(|| {
                    DynError::LimitExceeded("cell sample count overflow".to_owned())
                })?;
            }
            cell.channels = PersistentChannels::from_values(values);
            cell.sample_mass += sample_delta as f32;
            cell.confidence = 1.0;
            cell.canonicalize_zero();
            cell.validate()?;
            next.l2.insert(*index, cell);
        }
        next.l3 = derive_l3(&next.l2)?;
        next.validate(&self.limits)?;
        let next_snapshot = encode_snapshot(&next, &self.limits)?;

        self.state = next;
        self.snapshot_sha256 = format!("{:x}", Sha256::digest(&next_snapshot));
        self.client_life_epoch = batch.client_life_epoch;
        self.server_frame = batch.server_frame;
        self.last_event_id = previous_event_id.unwrap_or(baseline_event_id);
        self.life_event_count = next_life_events;
        self.accepted_event_count = next_accepted_events;
        Ok(DynIngestReport {
            events_applied: batch.events.len(),
            cells_updated: deposits.len(),
            decay_intervals,
            environment_steps: self.state.environment_steps(),
            client_life_epoch: self.client_life_epoch,
            server_frame: self.server_frame,
            last_event_id: self.last_event_id,
        })
    }

    /// Atomically ingests one public-event transaction and assembles that
    /// transaction's same-frame Dyn24 feature block.
    ///
    /// All work is staged on a clone. The live runtime is replaced only after
    /// both the exact event ingest and exact feature query succeed, so a bad
    /// query cannot consume event IDs or advance any runtime cursor.
    pub fn commit_frame(
        &mut self,
        batch: DynIngestBatch<'_>,
        query: DynRuntimeQuery,
    ) -> DynResult<(DynIngestReport, DynFeatureBlock)> {
        let mut candidate = self.clone();
        let report = candidate.ingest_events(batch)?;
        let features = candidate.feature_block(query)?;
        *self = candidate;
        Ok((report, features))
    }

    /// Low-level qualification/checkpoint primitive.
    ///
    /// Production frame assembly must use [`Self::commit_frame`] so the query
    /// and its event transaction have one commit boundary.
    pub fn feature_block(&self, query: DynRuntimeQuery) -> DynResult<DynFeatureBlock> {
        if query.map_epoch != self.state.fence().map_epoch {
            return Err(DynError::FenceMismatch("map_epoch"));
        }
        if query.client_id != self.state.client_id() {
            return Err(DynError::FenceMismatch("client_id"));
        }
        if query.client_life_epoch != self.client_life_epoch || self.client_life_epoch == 0 {
            return Err(DynError::StaleClientLifeEpoch {
                expected: self.client_life_epoch,
                found: query.client_life_epoch,
            });
        }
        if query.environment_steps != self.state.environment_steps() {
            return Err(DynError::StaleEnvironmentSteps {
                expected: self.state.environment_steps(),
                found: query.environment_steps,
            });
        }
        if query.server_frame != self.server_frame {
            return Err(DynError::StaleServerFrame {
                previous: self.server_frame,
                found: query.server_frame,
            });
        }
        if let Some(thermal) = query.thermal
            && (thermal.observed_tick > query.server_frame
                || query.server_frame - thermal.observed_tick > THERMAL_MAX_AGE_TICKS)
        {
            return Err(DynError::InvalidFormat(
                "thermal feature is future-dated or stale".to_owned(),
            ));
        }
        self.state.feature_block(DynFeatureInput {
            fence: self.state.fence(),
            world_position: query.world_position,
            yaw_degrees: query.yaw_degrees,
            thermal: query.thermal,
            survivability: query.survivability,
            search_radius: query.search_radius,
            score_scale: query.score_scale,
        })
    }
}

fn validate_runtime_cursor(
    client_life_epoch: u64,
    server_frame: u64,
    last_event_id: u64,
    life_event_count: u64,
    accepted_event_count: u64,
    limits: &DynLimits,
) -> DynResult<()> {
    if client_life_epoch == 0 {
        if server_frame != 0
            || last_event_id != 0
            || life_event_count != 0
            || accepted_event_count != 0
        {
            return Err(DynError::InvalidFormat(
                "uninitialized runtime checkpoint has a nonzero cursor".to_owned(),
            ));
        }
        return Ok(());
    }
    if server_frame == 0 {
        return Err(DynError::InvalidFormat(
            "initialized runtime checkpoint has server frame zero".to_owned(),
        ));
    }
    if life_event_count > limits.max_events_per_client_life {
        return Err(DynError::LimitExceeded(format!(
            "events per client life {life_event_count} > {}",
            limits.max_events_per_client_life
        )));
    }
    if accepted_event_count < life_event_count
        || (life_event_count == 0 && last_event_id != 0)
        || (life_event_count != 0 && last_event_id == 0)
    {
        return Err(DynError::InvalidFormat(
            "runtime checkpoint event cursor is inconsistent".to_owned(),
        ));
    }
    if last_event_id != 0 {
        let event_frame = last_event_id >> 3;
        let kind_code = last_event_id & 7;
        if event_frame == 0 || event_frame > server_frame || !(1..=5).contains(&kind_code) {
            return Err(DynError::InvalidFormat(
                "runtime checkpoint last event identity is malformed".to_owned(),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct RuntimeCheckpointHeader {
    snapshot_len: usize,
    snapshot_sha256: [u8; 32],
    fence: DynFence,
    client_id: u32,
    client_count: u32,
    environment_steps: u64,
    client_life_epoch: u64,
    server_frame: u64,
    last_event_id: u64,
    life_event_count: u64,
    accepted_event_count: u64,
}

impl RuntimeCheckpointHeader {
    fn decode(checkpoint: &[u8], limits: &DynLimits) -> DynResult<Self> {
        let maximum = RUNTIME_CHECKPOINT_HEADER_BYTES
            .checked_add(limits.max_compressed_snapshot_bytes)
            .ok_or_else(|| DynError::LimitExceeded("checkpoint limit overflow".to_owned()))?;
        if checkpoint.len() > maximum {
            return Err(DynError::LimitExceeded(format!(
                "runtime checkpoint bytes {} > {maximum}",
                checkpoint.len()
            )));
        }
        let mut reader = Reader::new(checkpoint);
        if reader.take(8)? != DYN_RUNTIME_CHECKPOINT_MAGIC {
            return Err(DynError::InvalidFormat(
                "invalid Dyn runtime checkpoint magic".to_owned(),
            ));
        }
        if reader.u16()? != DYN_RUNTIME_CHECKPOINT_VERSION {
            return Err(DynError::InvalidFormat(
                "unsupported Dyn runtime checkpoint version".to_owned(),
            ));
        }
        if reader.u16()? != 0 || reader.u32()? as usize != RUNTIME_CHECKPOINT_HEADER_BYTES {
            return Err(DynError::InvalidFormat(
                "invalid Dyn runtime checkpoint header".to_owned(),
            ));
        }
        let snapshot_len = reader.count(limits.max_compressed_snapshot_bytes, "snapshot bytes")?;
        let snapshot_sha256 = reader.array_32()?;
        let fence = DynFence {
            atlas_sha256: reader.array_32()?,
            map_sha256: reader.array_32()?,
            origin: AtlasOrigin([reader.i64()?, reader.i64()?, reader.i64()?]),
            map_epoch: reader.u64()?,
        };
        fence.validate()?;
        let result = Self {
            snapshot_len,
            snapshot_sha256,
            fence,
            client_id: reader.u32()?,
            client_count: reader.u32()?,
            environment_steps: reader.u64()?,
            client_life_epoch: reader.u64()?,
            server_frame: reader.u64()?,
            last_event_id: reader.u64()?,
            life_event_count: reader.u64()?,
            accepted_event_count: reader.u64()?,
        };
        validate_client_identity(result.client_id, result.client_count, limits)?;
        if reader.position() != RUNTIME_CHECKPOINT_HEADER_BYTES
            || checkpoint.len() != RUNTIME_CHECKPOINT_HEADER_BYTES + snapshot_len
        {
            return Err(DynError::InvalidFormat(
                "runtime checkpoint length/header mismatch".to_owned(),
            ));
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Signal {
    direction: [f32; 3],
    score: f32,
}

#[derive(Clone, Copy, Debug)]
struct CoarseCandidate {
    index: GridIndex,
    rank: f32,
}

#[derive(Clone, Copy, Debug)]
struct FineCandidate {
    index: GridIndex,
    world_center: [f64; 3],
    score: f32,
    rank: f32,
}

impl FineCandidate {
    fn better_than(self, other: Self) -> bool {
        self.rank > other.rank || (self.rank == other.rank && self.index < other.index)
    }
}

fn push_coarse_candidate(candidates: &mut Vec<CoarseCandidate>, candidate: CoarseCandidate) {
    let position = candidates
        .iter()
        .position(|current| {
            candidate.rank > current.rank
                || (candidate.rank == current.rank && candidate.index < current.index)
        })
        .unwrap_or(candidates.len());
    candidates.insert(position, candidate);
    candidates.truncate(COARSE_PARENT_BUDGET);
}

fn write_signal(values: &mut [f32; DYN_FEATURE_WIDTH], offset: usize, signal: Signal) {
    values[offset..offset + 3].copy_from_slice(&signal.direction);
    values[offset + 3] = signal.score;
}

fn signal_from_world_point(
    origin: [f64; 3],
    destination: [f64; 3],
    score: f32,
    yaw_degrees: f32,
    radius: f32,
) -> Signal {
    let delta = [
        (destination[0] - origin[0]) as f32,
        (destination[1] - origin[1]) as f32,
        (destination[2] - origin[2]) as f32,
    ];
    let yaw = yaw_degrees.to_radians();
    let (sin, cos) = yaw.sin_cos();
    let inverse_radius = radius.recip();
    Signal {
        direction: [
            ((delta[0] * cos + delta[1] * sin) * inverse_radius).clamp(-1.0, 1.0),
            ((delta[0] * sin - delta[1] * cos) * inverse_radius).clamp(-1.0, 1.0),
            (delta[2] * inverse_radius).clamp(-1.0, 1.0),
        ],
        score,
    }
}

fn normalize_score(value: f32, scale: f32) -> f32 {
    (value.max(0.0) / scale).tanh()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2) + (right[2] - left[2]).powi(2))
        .sqrt()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchExpectation {
    pub fence: DynFence,
    pub client_count: u32,
    pub environment_steps: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynBatchReport {
    pub compressed_bytes: usize,
    pub resident_bytes: usize,
    pub soft_limit_exceeded: bool,
}

#[derive(Clone, Debug)]
pub struct DynBatch {
    pub states: Vec<DynState>,
    pub report: DynBatchReport,
}

impl DynBatch {
    pub fn decode(
        snapshots: &[&[u8]],
        expectation: BatchExpectation,
        limits: &DynLimits,
    ) -> DynResult<Self> {
        expectation.fence.validate()?;
        validate_client_identity(0, expectation.client_count, limits)?;
        if snapshots.len() != expectation.client_count as usize {
            return Err(DynError::InvalidFormat(format!(
                "batch has {} snapshots, expected {}",
                snapshots.len(),
                expectation.client_count
            )));
        }
        let compressed_bytes = snapshots.iter().try_fold(0_usize, |total, snapshot| {
            total.checked_add(snapshot.len()).ok_or_else(|| {
                DynError::LimitExceeded("batch compressed byte count overflow".to_owned())
            })
        })?;
        if compressed_bytes > limits.batch_hard_compressed_bytes {
            return Err(DynError::LimitExceeded(format!(
                "batch compressed bytes {compressed_bytes} > hard limit {}",
                limits.batch_hard_compressed_bytes
            )));
        }

        let mut states = Vec::with_capacity(snapshots.len());
        let mut client_ids = BTreeSet::new();
        for snapshot in snapshots {
            let state = decode_snapshot(snapshot, expectation.fence, limits)?;
            if state.client_count != expectation.client_count {
                return Err(DynError::FenceMismatch("client_count"));
            }
            if state.environment_steps != expectation.environment_steps {
                return Err(DynError::StaleEnvironmentSteps {
                    expected: expectation.environment_steps,
                    found: state.environment_steps,
                });
            }
            if !client_ids.insert(state.client_id) {
                return Err(DynError::InvalidFormat(format!(
                    "duplicate Dyn client id {}",
                    state.client_id
                )));
            }
            states.push(state);
        }
        if client_ids.len() != expectation.client_count as usize
            || client_ids.iter().copied().ne(0..expectation.client_count)
        {
            return Err(DynError::InvalidFormat(
                "Dyn batch client IDs are not exactly 0..client_count".to_owned(),
            ));
        }
        states.sort_by_key(DynState::client_id);
        let resident_bytes = states.iter().try_fold(0_usize, |total, state| {
            total
                .checked_add(state.resident_bytes_estimate())
                .ok_or_else(|| {
                    DynError::LimitExceeded("batch resident byte count overflow".to_owned())
                })
        })?;
        if resident_bytes > limits.batch_hard_resident_bytes {
            return Err(DynError::LimitExceeded(format!(
                "batch resident bytes {resident_bytes} > hard limit {}",
                limits.batch_hard_resident_bytes
            )));
        }
        Ok(Self {
            states,
            report: DynBatchReport {
                compressed_bytes,
                resident_bytes,
                soft_limit_exceeded: compressed_bytes > limits.batch_soft_compressed_bytes,
            },
        })
    }
}

pub fn encode_snapshot(state: &DynState, limits: &DynLimits) -> DynResult<Vec<u8>> {
    state.validate(limits)?;
    let materialized = state
        .l2
        .len()
        .checked_add(state.l3.len())
        .ok_or_else(|| DynError::LimitExceeded("snapshot cell count overflow".to_owned()))?;
    let uncompressed_len = materialized
        .checked_mul(DYN_CELL_ENCODED_BYTES)
        .ok_or_else(|| DynError::LimitExceeded("snapshot byte count overflow".to_owned()))?;
    if uncompressed_len > limits.max_uncompressed_snapshot_bytes {
        return Err(DynError::LimitExceeded(format!(
            "uncompressed snapshot bytes {uncompressed_len} > {}",
            limits.max_uncompressed_snapshot_bytes
        )));
    }
    let mut payload = Vec::with_capacity(uncompressed_len);
    encode_cells(&mut payload, &state.l2);
    encode_cells(&mut payload, &state.l3);
    let payload_sha256: [u8; 32] = Sha256::digest(&payload).into();
    let compressed = zstd::stream::encode_all(Cursor::new(&payload), i32::from(ZSTD_LEVEL))?;
    let total_len = SNAPSHOT_HEADER_BYTES
        .checked_add(compressed.len())
        .ok_or_else(|| DynError::LimitExceeded("compressed snapshot size overflow".to_owned()))?;
    if total_len > limits.max_compressed_snapshot_bytes {
        return Err(DynError::LimitExceeded(format!(
            "compressed snapshot bytes {total_len} > {}",
            limits.max_compressed_snapshot_bytes
        )));
    }
    let mut output = Vec::with_capacity(total_len);
    output.extend_from_slice(DYN_MAGIC);
    push_u16(&mut output, DYN_SCHEMA_VERSION);
    push_u16(&mut output, BYTE_ORDER_LITTLE);
    push_u32(&mut output, SNAPSHOT_HEADER_BYTES as u32);
    output.push(COMPRESSION_ZSTD);
    output.push(ZSTD_LEVEL as u8);
    push_u16(&mut output, 0);
    push_u32(&mut output, 0);
    push_u64(&mut output, payload.len() as u64);
    push_u64(&mut output, compressed.len() as u64);
    output.extend_from_slice(&payload_sha256);
    output.extend_from_slice(&state.fence.atlas_sha256);
    output.extend_from_slice(&state.fence.map_sha256);
    for value in state.fence.origin.0 {
        push_i64(&mut output, value);
    }
    push_u32(&mut output, DYN_L2_CELL_SIZE);
    push_u32(&mut output, DYN_L3_CELL_SIZE);
    push_u64(&mut output, state.fence.map_epoch);
    push_u64(&mut output, state.environment_steps);
    push_u32(&mut output, state.client_id);
    push_u32(&mut output, state.client_count);
    push_u64(&mut output, state.l2.len() as u64);
    push_u64(&mut output, state.l3.len() as u64);
    debug_assert_eq!(output.len(), SNAPSHOT_HEADER_BYTES);
    output.extend_from_slice(&compressed);
    Ok(output)
}

pub fn decode_snapshot(
    snapshot: &[u8],
    expected_fence: DynFence,
    limits: &DynLimits,
) -> DynResult<DynState> {
    expected_fence.validate()?;
    let header = SnapshotHeader::decode(snapshot, limits)?;
    header.fence.check(expected_fence)?;
    let compressed = snapshot
        .get(SNAPSHOT_HEADER_BYTES..)
        .ok_or_else(|| DynError::InvalidFormat("truncated Dyn snapshot".to_owned()))?;
    let mut decoder = zstd::stream::read::Decoder::new(Cursor::new(compressed))?;
    let window_log = usize::BITS
        - limits
            .max_uncompressed_snapshot_bytes
            .max(1)
            .leading_zeros();
    decoder.window_log_max(window_log)?;
    let read_limit = (header.uncompressed_len as u64)
        .checked_add(1)
        .ok_or_else(|| DynError::LimitExceeded("uncompressed read limit overflow".to_owned()))?;
    let mut bounded = decoder.take(read_limit);
    let mut payload = Vec::with_capacity(header.uncompressed_len);
    bounded.read_to_end(&mut payload)?;
    if payload.len() != header.uncompressed_len {
        return Err(DynError::InvalidFormat(format!(
            "uncompressed Dyn length mismatch: expected {}, got {}",
            header.uncompressed_len,
            payload.len()
        )));
    }
    let actual_digest: [u8; 32] = Sha256::digest(&payload).into();
    if actual_digest != header.payload_sha256 {
        return Err(DynError::DigestMismatch);
    }
    let expected_payload_len = header
        .l2_count
        .checked_add(header.l3_count)
        .and_then(|count| count.checked_mul(DYN_CELL_ENCODED_BYTES))
        .ok_or_else(|| DynError::LimitExceeded("Dyn payload length overflow".to_owned()))?;
    if payload.len() != expected_payload_len {
        return Err(DynError::InvalidFormat(
            "Dyn cell counts do not match uncompressed payload".to_owned(),
        ));
    }
    let mut reader = Reader::new(&payload);
    let l2 = decode_cells(&mut reader, header.l2_count, "L2")?;
    let l3 = decode_cells(&mut reader, header.l3_count, "L3")?;
    reader.finish()?;
    let state = DynState {
        fence: header.fence,
        client_id: header.client_id,
        client_count: header.client_count,
        environment_steps: header.environment_steps,
        l2,
        l3,
    };
    state.validate(limits)?;
    Ok(state)
}

#[derive(Clone, Copy, Debug)]
struct SnapshotHeader {
    uncompressed_len: usize,
    payload_sha256: [u8; 32],
    fence: DynFence,
    environment_steps: u64,
    client_id: u32,
    client_count: u32,
    l2_count: usize,
    l3_count: usize,
}

impl SnapshotHeader {
    fn decode(snapshot: &[u8], limits: &DynLimits) -> DynResult<Self> {
        if snapshot.starts_with(RETIRED_DYN_MAGIC) {
            return Err(DynError::RetiredSchema);
        }
        if snapshot.len() > limits.max_compressed_snapshot_bytes {
            return Err(DynError::LimitExceeded(format!(
                "compressed snapshot bytes {} > {}",
                snapshot.len(),
                limits.max_compressed_snapshot_bytes
            )));
        }
        let mut reader = Reader::new(snapshot);
        if reader.take(8)? != DYN_MAGIC {
            return Err(DynError::InvalidFormat("invalid Dyn magic".to_owned()));
        }
        let schema = reader.u16()?;
        if schema != DYN_SCHEMA_VERSION {
            return Err(DynError::MixedSchema {
                expected: DYN_SCHEMA_VERSION,
                found: schema,
            });
        }
        if reader.u16()? != BYTE_ORDER_LITTLE {
            return Err(DynError::InvalidFormat(
                "Dyn byte order is not little-endian".to_owned(),
            ));
        }
        if reader.u32()? as usize != SNAPSHOT_HEADER_BYTES {
            return Err(DynError::InvalidFormat(
                "unexpected Dyn header size".to_owned(),
            ));
        }
        if reader.u8()? != COMPRESSION_ZSTD
            || reader.u8()? != ZSTD_LEVEL as u8
            || reader.u16()? != 0
            || reader.u32()? != 0
        {
            return Err(DynError::InvalidFormat(
                "unsupported Dyn compression or nonzero reserved field".to_owned(),
            ));
        }
        let uncompressed_len = reader.count(
            limits.max_uncompressed_snapshot_bytes,
            "uncompressed snapshot bytes",
        )?;
        let compressed_len = reader.count(
            limits
                .max_compressed_snapshot_bytes
                .saturating_sub(SNAPSHOT_HEADER_BYTES),
            "compressed payload bytes",
        )?;
        let payload_sha256 = reader.array_32()?;
        let fence = DynFence {
            atlas_sha256: reader.array_32()?,
            map_sha256: reader.array_32()?,
            origin: AtlasOrigin([reader.i64()?, reader.i64()?, reader.i64()?]),
            map_epoch: 0,
        };
        let l2_cell_size = reader.u32()?;
        let l3_cell_size = reader.u32()?;
        if l2_cell_size != DYN_L2_CELL_SIZE || l3_cell_size != DYN_L3_CELL_SIZE {
            return Err(DynError::InvalidFormat(format!(
                "Dyn cell-size fence mismatch: L2={l2_cell_size}, L3={l3_cell_size}"
            )));
        }
        let fence = DynFence {
            map_epoch: reader.u64()?,
            ..fence
        };
        fence.validate()?;
        let environment_steps = reader.u64()?;
        let client_id = reader.u32()?;
        let client_count = reader.u32()?;
        validate_client_identity(client_id, client_count, limits)?;
        let l2_count = reader.count(limits.max_l2_cells, "L2 cells")?;
        let l3_count = reader.count(limits.max_l3_cells, "L3 cells")?;
        let materialized = l2_count
            .checked_add(l3_count)
            .ok_or_else(|| DynError::LimitExceeded("materialized count overflow".to_owned()))?;
        if materialized > limits.max_materialized_cells {
            return Err(DynError::LimitExceeded(format!(
                "materialized cells {materialized} > {}",
                limits.max_materialized_cells
            )));
        }
        if reader.position() != SNAPSHOT_HEADER_BYTES
            || snapshot.len() != SNAPSHOT_HEADER_BYTES + compressed_len
        {
            return Err(DynError::InvalidFormat(
                "compressed Dyn length/header mismatch".to_owned(),
            ));
        }
        Ok(Self {
            uncompressed_len,
            payload_sha256,
            fence,
            environment_steps,
            client_id,
            client_count,
            l2_count,
            l3_count,
        })
    }
}

fn encode_cells(output: &mut Vec<u8>, cells: &BTreeMap<GridIndex, DynCell>) {
    for (index, cell) in cells {
        push_i32(output, index.x);
        push_i32(output, index.y);
        push_i32(output, index.z);
        for value in cell.channels.values() {
            push_f32(output, value);
        }
        push_f32(output, cell.sample_mass);
        push_f32(output, cell.confidence);
    }
}

fn decode_cells(
    reader: &mut Reader<'_>,
    count: usize,
    level: &str,
) -> DynResult<BTreeMap<GridIndex, DynCell>> {
    let mut cells = BTreeMap::new();
    let mut previous = None;
    for _ in 0..count {
        let index = GridIndex::new(reader.i32()?, reader.i32()?, reader.i32()?);
        if previous.is_some_and(|value| index <= value) {
            return Err(DynError::InvalidFormat(format!(
                "Dyn {level} cells are not strictly ordered (iz,iy,ix)"
            )));
        }
        previous = Some(index);
        let channels = PersistentChannels::from_values([
            reader.f32()?,
            reader.f32()?,
            reader.f32()?,
            reader.f32()?,
            reader.f32()?,
        ]);
        let cell = DynCell::new(channels, reader.f32()?, reader.f32()?)?;
        cells.insert(index, cell);
    }
    Ok(cells)
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(output: &mut Vec<u8>, value: i32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(output: &mut Vec<u8>, value: i64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(output: &mut Vec<u8>, value: f32) {
    output.extend_from_slice(&value.to_le_bytes());
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, length: usize) -> DynResult<&'a [u8]> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or_else(|| DynError::InvalidFormat("Dyn cursor overflow".to_owned()))?;
        let result = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| DynError::InvalidFormat("truncated Dyn payload".to_owned()))?;
        self.cursor = end;
        Ok(result)
    }

    fn position(self) -> usize {
        self.cursor
    }

    fn remaining(self) -> usize {
        self.bytes.len() - self.cursor
    }

    fn finish(self) -> DynResult<()> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(DynError::InvalidFormat(
                "trailing Dyn payload bytes".to_owned(),
            ))
        }
    }

    fn count(&mut self, maximum: usize, name: &str) -> DynResult<usize> {
        let count = usize::try_from(self.u64()?)
            .map_err(|_| DynError::LimitExceeded(format!("{name} exceeds usize")))?;
        if count > maximum {
            return Err(DynError::LimitExceeded(format!(
                "{name} {count} > {maximum}"
            )));
        }
        Ok(count)
    }

    fn u8(&mut self) -> DynResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> DynResult<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> DynResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> DynResult<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> DynResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> DynResult<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> DynResult<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn array_32(&mut self) -> DynResult<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }
}
