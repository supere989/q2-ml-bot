//! Hot-path kernels for q2-ml-bot's per-environment Vector Lattice.
//!
//! The policy contract retains a named 24-float Dyn block. Q2LAT002 Rust state
//! is authoritative for factual event deposits, deterministic decay, derived
//! L3, persistence, and feature assembly. Python only supplies identity-fenced
//! factual event tuples and query facts; it cannot inject packed cells or Dyn24
//! values into the live runtime.

use std::collections::BTreeMap;

pub mod atlas;
pub mod dynstate;

const CHANNELS: usize = 5;
const OUTPUTS: usize = 4;
const FEATURES: usize = 24;
const FEATURE_BUNDLE: usize = FEATURES + 1;
const STATE_MAGIC: &[u8; 8] = b"Q2LAT001";
pub const PACKED_CELL_WIDTH: usize = 9;
pub const SCORE_EVENT_WIDTH: usize = 11;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedCell {
    pub voxel: [i32; 3],
    pub raw_scores: [f32; CHANNELS],
    pub confidence: f32,
}

impl PackedCell {
    pub fn from_row(row: &[f32]) -> Option<Self> {
        if row.len() != PACKED_CELL_WIDTH {
            return None;
        }
        Some(Self {
            voxel: [row[0] as i32, row[1] as i32, row[2] as i32],
            raw_scores: [row[3], row[4], row[5], row[6], row[7]],
            confidence: row[8],
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct IndexedCell {
    packed: PackedCell,
    sample_mass: f32,
}

impl IndexedCell {
    fn from_packed(packed: PackedCell) -> Self {
        let confidence = packed.confidence.clamp(0.0, 1.0);
        let sample_mass = if confidence >= 1.0 {
            24.0
        } else {
            (confidence * 24.0_f32.ln_1p()).exp_m1()
        };
        Self {
            packed,
            sample_mass,
        }
    }

    fn apply_score_event(
        &mut self,
        score_deltas: [f32; CHANNELS],
        sample_delta: f32,
        force_confident: bool,
        confidence_override: Option<f32>,
    ) {
        for (score, delta) in self.packed.raw_scores.iter_mut().zip(score_deltas) {
            *score += delta;
        }
        self.sample_mass = (self.sample_mass + sample_delta).max(0.0);
        let learned = self.sample_mass.ln_1p() / 24.0_f32.ln_1p();
        self.packed.confidence = if let Some(confidence) = confidence_override {
            confidence.clamp(0.0, 1.0)
        } else if force_confident {
            1.0
        } else {
            self.packed.confidence.max(learned.min(1.0))
        };
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    rank: f32,
    score: f32,
    center: [f32; 3],
    found: bool,
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            rank: -1.0,
            score: 0.0,
            center: [0.0; 3],
            found: false,
        }
    }
}

/// Return `[dx, dy, dz, score]` for engagement, threat, opportunity,
/// self-fire, and deaths, matching `VoxelSpatialReward` channel order.
pub fn nearest_signals(
    cells: &[PackedCell],
    position: [f32; 3],
    search_radius: f32,
    voxel_size: f32,
    score_scale: f32,
) -> [[f32; OUTPUTS]; CHANNELS] {
    nearest_signals_iter(
        cells.iter(),
        position,
        search_radius,
        voxel_size,
        score_scale,
    )
}

fn nearest_signals_iter<'a>(
    cells: impl Iterator<Item = &'a PackedCell>,
    position: [f32; 3],
    search_radius: f32,
    voxel_size: f32,
    score_scale: f32,
) -> [[f32; OUTPUTS]; CHANNELS] {
    let radius = search_radius.max(voxel_size).max(1.0);
    let voxel = voxel_size.max(1.0);
    let scale = score_scale.max(0.1);
    let mut best = [Candidate::default(); CHANNELS];

    for cell in cells {
        let center = [
            (cell.voxel[0] as f32 + 0.5) * voxel,
            (cell.voxel[1] as f32 + 0.5) * voxel,
            (cell.voxel[2] as f32 + 0.5) * voxel,
        ];
        let delta = [
            center[0] - position[0],
            center[1] - position[1],
            center[2] - position[2],
        ];
        let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if distance > radius {
            continue;
        }
        let distance_scale = (distance / voxel).max(1.0);
        for (channel, candidate) in best.iter_mut().enumerate() {
            let raw = cell.raw_scores[channel];
            if raw <= 0.0 {
                continue;
            }
            let score = (raw / scale).tanh() * cell.confidence;
            let rank = score / distance_scale;
            if rank > candidate.rank {
                *candidate = Candidate {
                    rank,
                    score,
                    center,
                    found: true,
                };
            }
        }
    }

    let mut output = [[0.0; OUTPUTS]; CHANNELS];
    for channel in 0..CHANNELS {
        let candidate = best[channel];
        if !candidate.found {
            continue;
        }
        output[channel] = [
            ((candidate.center[0] - position[0]) / radius).clamp(-1.0, 1.0),
            ((candidate.center[1] - position[1]) / radius).clamp(-1.0, 1.0),
            ((candidate.center[2] - position[2]) / radius).clamp(-1.0, 1.0),
            candidate.score,
        ];
    }
    output
}

/// Deterministic, incrementally updated per-environment lattice storage.
///
/// A BTreeMap makes persistence and equal-input query order stable across
/// processes. Python sends only cells changed by the current transition;
/// queries never rebuild or copy the complete lattice across the FFI boundary.
#[derive(Clone, Debug)]
pub struct LatticeIndex {
    cells: BTreeMap<[i32; 3], IndexedCell>,
    search_radius: f32,
    voxel_size: f32,
    score_scale: f32,
}

impl LatticeIndex {
    pub fn new(search_radius: f32, voxel_size: f32, score_scale: f32) -> Self {
        Self {
            cells: BTreeMap::new(),
            search_radius: search_radius.max(voxel_size).max(1.0),
            voxel_size: voxel_size.max(1.0),
            score_scale: score_scale.max(0.1),
        }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn upsert(&mut self, cell: PackedCell) {
        self.cells
            .insert(cell.voxel, IndexedCell::from_packed(cell));
    }

    pub fn apply_score_event(
        &mut self,
        voxel: [i32; 3],
        score_deltas: [f32; CHANNELS],
        sample_delta: f32,
        force_confident: bool,
        confidence_override: Option<f32>,
    ) {
        let cell = self.cells.entry(voxel).or_insert_with(|| {
            IndexedCell::from_packed(PackedCell {
                voxel,
                raw_scores: [0.0; CHANNELS],
                confidence: 0.0,
            })
        });
        cell.apply_score_event(
            score_deltas,
            sample_delta,
            force_confident,
            confidence_override,
        );
    }

    pub fn remove(&mut self, voxel: [i32; 3]) -> bool {
        self.cells.remove(&voxel).is_some()
    }

    pub fn nearest_signals(&self, position: [f32; 3]) -> [[f32; OUTPUTS]; CHANNELS] {
        nearest_signals_iter(
            self.cells.values().map(|cell| &cell.packed),
            position,
            self.search_radius,
            self.voxel_size,
            self.score_scale,
        )
    }

    /// Produce the existing 24-float policy tail without changing its shape.
    /// Survivability remains supplied by Python until its combat-state tracker
    /// moves into Rust; the lattice owns slots 0..20.
    pub fn features(
        &self,
        position: [f32; 3],
        current_voxel: [i32; 3],
        survivability: [f32; 3],
    ) -> [f32; FEATURES] {
        let bundle = self.feature_bundle(position, current_voxel, survivability);
        bundle[..FEATURES].try_into().unwrap()
    }

    /// Policy features plus nearest-death score in slot 24 for reward shaping.
    pub fn feature_bundle(
        &self,
        position: [f32; 3],
        current_voxel: [i32; 3],
        survivability: [f32; 3],
    ) -> [f32; FEATURE_BUNDLE] {
        let mut output = [0.0; FEATURE_BUNDLE];
        if let Some(current) = self.cells.get(&current_voxel).map(|cell| &cell.packed) {
            for (index, raw) in current.raw_scores[..4].iter().enumerate() {
                output[index] = (raw.max(0.0) / self.score_scale).tanh();
            }
            output[4] = current.confidence;
        }
        let nearest = self.nearest_signals(position);
        for (channel, values) in nearest.iter().take(4).enumerate() {
            let offset = 5 + channel * OUTPUTS;
            output[offset..offset + OUTPUTS].copy_from_slice(values);
        }
        output[21..24].copy_from_slice(&survivability);
        output[24] = nearest[4][3];
        output
    }

    /// Stable little-endian snapshot. Dynamic route overlays are just cells
    /// to this layer; Python decides whether to omit them before applying.
    pub fn dumps(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24 + self.cells.len() * 36);
        bytes.extend_from_slice(STATE_MAGIC);
        bytes.extend_from_slice(&(self.cells.len() as u64).to_le_bytes());
        for cell in self.cells.values().map(|cell| &cell.packed) {
            for value in cell.voxel {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in cell.raw_scores {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.extend_from_slice(&cell.confidence.to_le_bytes());
        }
        bytes
    }

    pub fn loads(
        bytes: &[u8],
        search_radius: f32,
        voxel_size: f32,
        score_scale: f32,
    ) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[..8] != STATE_MAGIC {
            return Err("invalid lattice snapshot header".to_owned());
        }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let expected = 16usize
            .checked_add(
                count
                    .checked_mul(36)
                    .ok_or("lattice snapshot is too large")?,
            )
            .ok_or("lattice snapshot is too large")?;
        if bytes.len() != expected {
            return Err(format!(
                "invalid lattice snapshot length: expected {expected}, got {}",
                bytes.len()
            ));
        }
        let mut index = Self::new(search_radius, voxel_size, score_scale);
        let mut cursor = 16;
        for _ in 0..count {
            let mut voxel = [0; 3];
            for value in &mut voxel {
                *value = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
                cursor += 4;
            }
            let mut raw_scores = [0.0; CHANNELS];
            for value in &mut raw_scores {
                *value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
                cursor += 4;
            }
            let confidence = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            index.upsert(PackedCell {
                voxel,
                raw_scores,
                confidence,
            });
        }
        Ok(index)
    }
}

#[cfg(feature = "python")]
mod python {
    use super::{LatticeIndex, PACKED_CELL_WIDTH, PackedCell, SCORE_EVENT_WIDTH, nearest_signals};
    use crate::atlas::AtlasOrigin;
    use crate::atlas::{
        AtlasLimits, AtlasRuntime, ObjectiveBelief, RECOVERY_EVIDENCE_SCHEMA, RecoveryOverlay,
        RecoveryQuery, advisory_spatial_feature_names,
    };
    use crate::dynstate::{
        DYN_EVENT_NAMES, DYN_EVENT_SCHEMA, DYN_FEATURE_NAMES, DYN_FEATURE_SCHEMA_SHA256,
        DynEventKind, DynFence, DynIngestBatch, DynIngestReport, DynLimits, DynNamedEvent,
        DynResult, DynRuntime, DynRuntimeQuery, ThermalSignal,
    };
    use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict};
    use std::collections::{BTreeMap, BTreeSet};

    fn packed_rows(cells: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<PackedCell>> {
        let view = cells.as_array();
        if view.shape().len() != 2 || view.shape()[1] != PACKED_CELL_WIDTH {
            return Err(PyValueError::new_err(format!(
                "cells must have shape (N, {PACKED_CELL_WIDTH})"
            )));
        }
        let values = cells
            .as_slice()
            .map_err(|_| PyValueError::new_err("cells must be a C-contiguous float32 array"))?;
        Ok(values
            .chunks_exact(PACKED_CELL_WIDTH)
            .map(|row| PackedCell::from_row(row).expect("chunk width was validated"))
            .collect())
    }

    #[pyfunction(name = "nearest_signals")]
    fn nearest_signals_py<'py>(
        py: Python<'py>,
        cells: PyReadonlyArray2<'py, f32>,
        position: [f32; 3],
        search_radius: f32,
        voxel_size: f32,
        score_scale: f32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let packed = packed_rows(cells)?;
        let result = nearest_signals(&packed, position, search_radius, voxel_size, score_scale);
        let rows: Vec<Vec<f32>> = result.into_iter().map(Vec::from).collect();
        PyArray2::from_vec2(py, &rows).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyclass(name = "LatticeIndex", module = "q2_lattice_rs")]
    struct PyLatticeIndex {
        inner: LatticeIndex,
    }

    #[pyclass(name = "AtlasRuntime", module = "q2_lattice_rs")]
    struct PyAtlasRuntime {
        inner: AtlasRuntime,
    }

    #[pyclass(name = "DynRuntime", module = "q2_lattice_rs")]
    struct PyDynRuntime {
        inner: DynRuntime,
    }

    fn dyn_ingest_report<'py>(
        py: Python<'py>,
        report: DynIngestReport,
        snapshot_sha256: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("schema", DYN_EVENT_SCHEMA)?;
        result.set_item("events_applied", report.events_applied)?;
        result.set_item("cells_updated", report.cells_updated)?;
        result.set_item("decay_intervals", report.decay_intervals)?;
        result.set_item("environment_steps", report.environment_steps)?;
        result.set_item("client_life_epoch", report.client_life_epoch)?;
        result.set_item("server_frame", report.server_frame)?;
        result.set_item("last_event_id", report.last_event_id)?;
        result.set_item("snapshot_sha256", snapshot_sha256)?;
        Ok(result)
    }

    #[pymethods]
    impl PyDynRuntime {
        #[new]
        #[allow(clippy::too_many_arguments)]
        fn new(
            snapshot: &[u8],
            atlas_sha256: &str,
            map_sha256: &str,
            origin: [i64; 3],
            map_epoch: u64,
            client_id: u32,
            client_count: u32,
            environment_steps: u64,
        ) -> PyResult<Self> {
            let fence = DynFence {
                atlas_sha256: digest_from_hex(atlas_sha256, "Atlas SHA-256")?,
                map_sha256: digest_from_hex(map_sha256, "map SHA-256")?,
                origin: AtlasOrigin(origin),
                map_epoch,
            };
            let inner = DynRuntime::from_snapshot(
                snapshot,
                fence,
                client_id,
                client_count,
                environment_steps,
                &DynLimits::default(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(Self { inner })
        }

        #[staticmethod]
        #[allow(clippy::too_many_arguments)]
        fn from_checkpoint(
            checkpoint: &[u8],
            checkpoint_sha256: &str,
            atlas_sha256: &str,
            map_sha256: &str,
            origin: [i64; 3],
            map_epoch: u64,
            client_id: u32,
            client_count: u32,
            environment_steps: u64,
            client_life_epoch: u64,
            server_frame: u64,
        ) -> PyResult<Self> {
            let fence = DynFence {
                atlas_sha256: digest_from_hex(atlas_sha256, "Atlas SHA-256")?,
                map_sha256: digest_from_hex(map_sha256, "map SHA-256")?,
                origin: AtlasOrigin(origin),
                map_epoch,
            };
            let inner = DynRuntime::from_checkpoint(
                checkpoint,
                digest_from_hex(checkpoint_sha256, "checkpoint SHA-256")?,
                fence,
                client_id,
                client_count,
                environment_steps,
                client_life_epoch,
                server_frame,
                &DynLimits::default(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(Self { inner })
        }

        #[getter]
        fn snapshot_sha256(&self) -> &str {
            self.inner.snapshot_sha256()
        }

        #[getter]
        fn atlas_sha256(&self) -> String {
            digest_hex(&self.inner.state().fence().atlas_sha256)
        }

        #[getter]
        fn map_sha256(&self) -> String {
            digest_hex(&self.inner.state().fence().map_sha256)
        }

        #[getter]
        fn origin(&self) -> [i64; 3] {
            self.inner.state().fence().origin.0
        }

        #[getter]
        fn map_epoch(&self) -> u64 {
            self.inner.state().fence().map_epoch
        }

        #[getter]
        fn client_id(&self) -> u32 {
            self.inner.state().client_id()
        }

        #[getter]
        fn client_count(&self) -> u32 {
            self.inner.state().client_count()
        }

        #[getter]
        fn environment_steps(&self) -> u64 {
            self.inner.state().environment_steps()
        }

        #[getter]
        fn client_life_epoch(&self) -> u64 {
            self.inner.client_life_epoch()
        }

        #[getter]
        fn server_frame(&self) -> u64 {
            self.inner.server_frame()
        }

        #[getter]
        fn last_event_id(&self) -> u64 {
            self.inner.last_event_id()
        }

        #[getter]
        fn accepted_event_count(&self) -> u64 {
            self.inner.accepted_event_count()
        }

        #[getter]
        fn checkpoint_sha256(&self) -> PyResult<String> {
            self.inner
                .checkpoint_sha256()
                .map_err(|error| PyValueError::new_err(error.to_string()))
        }

        fn snapshot_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let bytes = self
                .inner
                .snapshot_bytes()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(PyBytes::new(py, &bytes))
        }

        fn checkpoint_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let bytes = self
                .inner
                .checkpoint_bytes()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(PyBytes::new(py, &bytes))
        }

        /// Qualification/checkpoint primitive. Production frame assembly uses
        /// `commit_frame`, because split ingest/query calls are not atomic.
        #[allow(clippy::too_many_arguments)]
        fn ingest_events<'py>(
            &mut self,
            py: Python<'py>,
            atlas_sha256: &str,
            map_sha256: &str,
            origin: [i64; 3],
            map_epoch: u64,
            client_id: u32,
            client_life_epoch: u64,
            server_frame: u64,
            expected_environment_steps: u64,
            environment_steps: u64,
            events: Vec<(u64, String, [f64; 3])>,
        ) -> PyResult<Bound<'py, PyDict>> {
            let fence = DynFence {
                atlas_sha256: digest_from_hex(atlas_sha256, "Atlas SHA-256")?,
                map_sha256: digest_from_hex(map_sha256, "map SHA-256")?,
                origin: AtlasOrigin(origin),
                map_epoch,
            };
            let events = events
                .into_iter()
                .map(|(event_id, kind, world_position)| {
                    Ok(DynNamedEvent {
                        event_id,
                        kind: DynEventKind::parse(&kind)?,
                        world_position,
                    })
                })
                .collect::<DynResult<Vec<_>>>()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let report = self
                .inner
                .ingest_events(DynIngestBatch {
                    fence,
                    client_id,
                    client_life_epoch,
                    server_frame,
                    expected_environment_steps,
                    environment_steps,
                    events: &events,
                })
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            dyn_ingest_report(py, report, self.inner.snapshot_sha256())
        }

        #[pyo3(signature = (
            atlas_sha256,
            map_sha256,
            origin,
            map_epoch,
            client_id,
            client_life_epoch,
            server_frame,
            expected_environment_steps,
            environment_steps,
            events,
            position,
            yaw_degrees,
            survivability,
            thermal=None,
            search_radius=2048.0,
            score_scale=8.0
        ))]
        #[allow(clippy::too_many_arguments)]
        fn commit_frame<'py>(
            &mut self,
            py: Python<'py>,
            atlas_sha256: &str,
            map_sha256: &str,
            origin: [i64; 3],
            map_epoch: u64,
            client_id: u32,
            client_life_epoch: u64,
            server_frame: u64,
            expected_environment_steps: u64,
            environment_steps: u64,
            events: Vec<(u64, String, [f64; 3])>,
            position: [f64; 3],
            yaw_degrees: f32,
            survivability: [f32; 3],
            thermal: Option<(u64, [f64; 3], f32, u64)>,
            search_radius: f32,
            score_scale: f32,
        ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArray1<f32>>, f32)> {
            let fence = DynFence {
                atlas_sha256: digest_from_hex(atlas_sha256, "Atlas SHA-256")?,
                map_sha256: digest_from_hex(map_sha256, "map SHA-256")?,
                origin: AtlasOrigin(origin),
                map_epoch,
            };
            let events = events
                .into_iter()
                .map(|(event_id, kind, world_position)| {
                    Ok(DynNamedEvent {
                        event_id,
                        kind: DynEventKind::parse(&kind)?,
                        world_position,
                    })
                })
                .collect::<DynResult<Vec<_>>>()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let thermal =
                thermal.map(
                    |(target_id, world_point, heat, observed_tick)| ThermalSignal {
                        target_id,
                        world_point,
                        heat,
                        observed_tick,
                    },
                );
            let (report, block) = self
                .inner
                .commit_frame(
                    DynIngestBatch {
                        fence,
                        client_id,
                        client_life_epoch,
                        server_frame,
                        expected_environment_steps,
                        environment_steps,
                        events: &events,
                    },
                    DynRuntimeQuery {
                        map_epoch,
                        client_id,
                        client_life_epoch,
                        environment_steps,
                        server_frame,
                        world_position: position,
                        yaw_degrees,
                        thermal,
                        survivability,
                        search_radius,
                        score_scale,
                    },
                )
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let result = dyn_ingest_report(py, report, self.inner.snapshot_sha256())?;
            let features = PyArray1::from_slice(py, &block.values);
            Ok((result, features, block.nearest_death_score))
        }

        #[pyo3(signature = (
            position,
            yaw_degrees,
            map_epoch,
            client_id,
            client_life_epoch,
            environment_steps,
            server_frame,
            survivability,
            thermal=None,
            search_radius=2048.0,
            score_scale=8.0
        ))]
        /// Qualification/checkpoint primitive. Production frame assembly uses
        /// `commit_frame`, because split ingest/query calls are not atomic.
        #[allow(clippy::too_many_arguments)]
        fn feature_block<'py>(
            &self,
            py: Python<'py>,
            position: [f64; 3],
            yaw_degrees: f32,
            map_epoch: u64,
            client_id: u32,
            client_life_epoch: u64,
            environment_steps: u64,
            server_frame: u64,
            survivability: [f32; 3],
            thermal: Option<(u64, [f64; 3], f32, u64)>,
            search_radius: f32,
            score_scale: f32,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let thermal =
                thermal.map(
                    |(target_id, world_point, heat, observed_tick)| ThermalSignal {
                        target_id,
                        world_point,
                        heat,
                        observed_tick,
                    },
                );
            let block = py
                .detach(|| {
                    self.inner.feature_block(DynRuntimeQuery {
                        map_epoch,
                        client_id,
                        client_life_epoch,
                        environment_steps,
                        server_frame,
                        world_position: position,
                        yaw_degrees,
                        thermal,
                        survivability,
                        search_radius,
                        score_scale,
                    })
                })
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(PyArray1::from_slice(py, &block.values))
        }
    }

    fn digest_from_hex(value: &str, label: &str) -> PyResult<[u8; 32]> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(PyValueError::new_err(format!(
                "{label} must contain exactly 64 lowercase hexadecimal characters"
            )));
        }
        let mut result = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            result[index] = u8::from_str_radix(
                std::str::from_utf8(pair).expect("validated ASCII hexadecimal"),
                16,
            )
            .map_err(|_| PyValueError::new_err(format!("{label} is invalid")))?;
        }
        Ok(result)
    }

    fn digest_hex(value: &[u8; 32]) -> String {
        value.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[pymethods]
    impl PyAtlasRuntime {
        #[new]
        #[allow(clippy::too_many_arguments)]
        fn new(
            manifest: &[u8],
            artifact_name: &str,
            atlas: &[u8],
            objective_artifact_name: &str,
            objectives: &[u8],
            bsp: &[u8],
            expected_map_id: &str,
            map_epoch: u64,
        ) -> PyResult<Self> {
            let inner = AtlasRuntime::from_bytes(
                manifest,
                artifact_name,
                atlas,
                objective_artifact_name,
                objectives,
                bsp,
                expected_map_id,
                map_epoch,
                &AtlasLimits::default(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(Self { inner })
        }

        #[getter]
        fn map_id(&self) -> &str {
            &self.inner.manifest().bsp.canonical_map_id
        }

        #[getter]
        fn map_epoch(&self) -> u64 {
            self.inner.map_epoch()
        }

        #[getter]
        fn atlas_sha256(&self) -> &str {
            self.inner.atlas_sha256()
        }

        #[getter]
        fn atlas_manifest_sha256(&self) -> &str {
            self.inner.manifest_sha256()
        }

        #[getter]
        fn objective_sha256(&self) -> &str {
            self.inner.objective_sha256()
        }

        #[getter]
        fn resident_bytes(&self) -> usize {
            self.inner.resident_bytes_estimate()
        }

        fn query_counters(&self) -> (u64, u64, u64, u64, u64) {
            let counters = self.inner.query_counters();
            (
                counters.accepted_queries,
                counters.atlas_lookup_ns,
                counters.recovery_ns,
                counters.guide_ns,
                counters.total_ns,
            )
        }

        #[pyo3(signature = (
            position,
            yaw_degrees,
            map_epoch,
            blocked_nodes=Vec::new(),
            dynamic_penalties=Vec::new(),
            enabled_mover_blockers=Vec::new(),
            time_to_impact_seconds=None
        ))]
        #[allow(clippy::too_many_arguments)]
        fn recovery_features<'py>(
            &self,
            py: Python<'py>,
            position: [f64; 3],
            yaw_degrees: f32,
            map_epoch: u64,
            blocked_nodes: Vec<[i32; 3]>,
            dynamic_penalties: Vec<([i32; 3], u32)>,
            enabled_mover_blockers: Vec<u32>,
            time_to_impact_seconds: Option<f32>,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let overlay =
                recovery_overlay(blocked_nodes, dynamic_penalties, enabled_mover_blockers);
            let block = py
                .detach(|| {
                    self.inner.recovery(
                        map_epoch,
                        RecoveryQuery {
                            world_position: position,
                            yaw_degrees,
                            overlay: &overlay,
                            time_to_impact_seconds,
                        },
                    )
                })
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(PyArray1::from_slice(py, &block.values))
        }

        #[pyo3(signature = (
            position,
            yaw_degrees,
            map_epoch,
            client_id,
            client_epoch,
            server_frame,
            blocked_nodes=Vec::new(),
            dynamic_penalties=Vec::new(),
            enabled_mover_blockers=Vec::new(),
            time_to_impact_seconds=None
        ))]
        #[allow(clippy::too_many_arguments)]
        fn recovery_features_with_evidence<'py>(
            &self,
            py: Python<'py>,
            position: [f64; 3],
            yaw_degrees: f32,
            map_epoch: u64,
            client_id: u32,
            client_epoch: u64,
            server_frame: u64,
            blocked_nodes: Vec<[i32; 3]>,
            dynamic_penalties: Vec<([i32; 3], u32)>,
            enabled_mover_blockers: Vec<u32>,
            time_to_impact_seconds: Option<f32>,
        ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyDict>)> {
            if client_epoch == 0 {
                return Err(PyValueError::new_err(
                    "recovery evidence client_epoch must be nonzero",
                ));
            }
            let overlay =
                recovery_overlay(blocked_nodes, dynamic_penalties, enabled_mover_blockers);
            let block = py
                .detach(|| {
                    self.inner.recovery(
                        map_epoch,
                        RecoveryQuery {
                            world_position: position,
                            yaw_degrees,
                            overlay: &overlay,
                            time_to_impact_seconds,
                        },
                    )
                })
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let evidence = PyDict::new(py);
            evidence.set_item("schema", RECOVERY_EVIDENCE_SCHEMA)?;
            evidence.set_item("atlas_sha256", self.inner.atlas_sha256())?;
            evidence.set_item("map_epoch", self.inner.map_epoch())?;
            evidence.set_item("client_id", client_id)?;
            evidence.set_item("client_epoch", client_epoch)?;
            evidence.set_item("server_frame", server_frame)?;
            evidence.set_item(
                "l1_index",
                [
                    block.evidence.l1_index.x,
                    block.evidence.l1_index.y,
                    block.evidence.l1_index.z,
                ],
            )?;
            evidence.set_item("cost_to_safety_q8", block.evidence.cost_to_safety_q8)?;
            evidence.set_item(
                "signed_safe_clearance_q8",
                block.evidence.signed_safe_clearance_q8,
            )?;
            evidence.set_item("hazard_types", block.evidence.hazard_types)?;
            evidence.set_item("hazard_severity", block.evidence.hazard_severity)?;
            evidence.set_item("atlas_region_id", block.evidence.atlas_region_id)?;
            evidence.set_item("hazard_component_id", block.evidence.hazard_component_id)?;
            evidence.set_item(
                "hazard_component_epoch",
                block
                    .evidence
                    .hazard_component_epoch(self.inner.map_epoch()),
            )?;
            evidence.set_item("confidence", block.evidence.confidence)?;
            Ok((PyArray1::from_slice(py, &block.values), evidence))
        }

        fn guide_features<'py>(
            &self,
            py: Python<'py>,
            position: [f64; 3],
            yaw_degrees: f32,
            map_epoch: u64,
            beliefs: Vec<(u32, f32)>,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let beliefs = objective_beliefs(beliefs);
            let block = py
                .detach(|| self.inner.guide(map_epoch, position, yaw_degrees, &beliefs))
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(PyArray1::from_slice(py, &block.values))
        }

        #[pyo3(signature = (
            position,
            yaw_degrees,
            map_epoch,
            beliefs,
            blocked_nodes=Vec::new(),
            dynamic_penalties=Vec::new(),
            enabled_mover_blockers=Vec::new(),
            time_to_impact_seconds=None
        ))]
        #[allow(clippy::too_many_arguments)]
        fn advisory_spatial_features<'py>(
            &self,
            py: Python<'py>,
            position: [f64; 3],
            yaw_degrees: f32,
            map_epoch: u64,
            beliefs: Vec<(u32, f32)>,
            blocked_nodes: Vec<[i32; 3]>,
            dynamic_penalties: Vec<([i32; 3], u32)>,
            enabled_mover_blockers: Vec<u32>,
            time_to_impact_seconds: Option<f32>,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let beliefs = objective_beliefs(beliefs);
            let overlay =
                recovery_overlay(blocked_nodes, dynamic_penalties, enabled_mover_blockers);
            let values = py
                .detach(|| {
                    self.inner.advisory_spatial_features(
                        map_epoch,
                        RecoveryQuery {
                            world_position: position,
                            yaw_degrees,
                            overlay: &overlay,
                            time_to_impact_seconds,
                        },
                        &beliefs,
                    )
                })
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(PyArray1::from_slice(py, &values))
        }
    }

    fn recovery_overlay(
        blocked_nodes: Vec<[i32; 3]>,
        dynamic_penalties: Vec<([i32; 3], u32)>,
        enabled_mover_blockers: Vec<u32>,
    ) -> RecoveryOverlay {
        RecoveryOverlay {
            blocked_nodes: blocked_nodes
                .into_iter()
                .map(|value| crate::atlas::GridIndex::new(value[0], value[1], value[2]))
                .collect::<BTreeSet<_>>(),
            dynamic_penalty_q8: dynamic_penalties
                .into_iter()
                .map(|(value, penalty)| {
                    (
                        crate::atlas::GridIndex::new(value[0], value[1], value[2]),
                        penalty,
                    )
                })
                .collect::<BTreeMap<_, _>>(),
            enabled_mover_blockers: enabled_mover_blockers.into_iter().collect(),
        }
    }

    fn objective_beliefs(values: Vec<(u32, f32)>) -> Vec<ObjectiveBelief> {
        values
            .into_iter()
            .map(|(objective_id, availability_belief)| ObjectiveBelief {
                objective_id,
                availability_belief,
            })
            .collect()
    }

    #[pyfunction]
    fn atlas_advisory_feature_names() -> Vec<&'static str> {
        advisory_spatial_feature_names()
    }

    #[pyfunction]
    fn dyn_feature_names() -> Vec<&'static str> {
        DYN_FEATURE_NAMES.to_vec()
    }

    #[pyfunction]
    fn dyn_feature_schema_sha256() -> &'static str {
        DYN_FEATURE_SCHEMA_SHA256
    }

    #[pyfunction]
    fn dyn_event_names() -> Vec<&'static str> {
        DYN_EVENT_NAMES.to_vec()
    }

    #[pyfunction]
    fn dyn_event_schema() -> &'static str {
        DYN_EVENT_SCHEMA
    }

    #[pymethods]
    impl PyLatticeIndex {
        #[new]
        fn new(search_radius: f32, voxel_size: f32, score_scale: f32) -> Self {
            Self {
                inner: LatticeIndex::new(search_radius, voxel_size, score_scale),
            }
        }

        fn __len__(&self) -> usize {
            self.inner.len()
        }

        fn clear(&mut self) {
            self.inner.clear();
        }

        fn remove(&mut self, voxel: [i32; 3]) -> bool {
            self.inner.remove(voxel)
        }

        fn upsert(&mut self, voxel: [i32; 3], raw_scores: [f32; 5], confidence: f32) {
            self.inner.upsert(PackedCell {
                voxel,
                raw_scores,
                confidence,
            });
        }

        fn apply_packed(&mut self, cells: PyReadonlyArray2<'_, f32>) -> PyResult<usize> {
            let packed = packed_rows(cells)?;
            let count = packed.len();
            for cell in packed {
                self.inner.upsert(cell);
            }
            Ok(count)
        }

        /// Rows are voxel xyz, five raw-score deltas, sample delta, and a
        /// force-confidence flag. Multiple Python events are coalesced first.
        fn apply_score_events(&mut self, events: PyReadonlyArray2<'_, f32>) -> PyResult<usize> {
            let view = events.as_array();
            if view.shape().len() != 2 || view.shape()[1] != SCORE_EVENT_WIDTH {
                return Err(PyValueError::new_err(format!(
                    "events must have shape (N, {SCORE_EVENT_WIDTH})"
                )));
            }
            let values = events.as_slice().map_err(|_| {
                PyValueError::new_err("events must be a C-contiguous float32 array")
            })?;
            let mut count = 0;
            for row in values.chunks_exact(SCORE_EVENT_WIDTH) {
                self.inner.apply_score_event(
                    [row[0] as i32, row[1] as i32, row[2] as i32],
                    [row[3], row[4], row[5], row[6], row[7]],
                    row[8],
                    row[9] > 0.5,
                    row[10].is_finite().then_some(row[10]),
                );
                count += 1;
            }
            Ok(count)
        }

        fn nearest_signals<'py>(
            &self,
            py: Python<'py>,
            position: [f32; 3],
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let result = py.detach(|| self.inner.nearest_signals(position));
            let rows: Vec<Vec<f32>> = result.into_iter().map(Vec::from).collect();
            PyArray2::from_vec2(py, &rows).map_err(|error| PyValueError::new_err(error.to_string()))
        }

        fn features<'py>(
            &self,
            py: Python<'py>,
            position: [f32; 3],
            current_voxel: [i32; 3],
            survivability: [f32; 3],
        ) -> Bound<'py, PyArray1<f32>> {
            let result = py.detach(|| self.inner.features(position, current_voxel, survivability));
            PyArray1::from_slice(py, &result)
        }

        fn feature_bundle<'py>(
            &self,
            py: Python<'py>,
            position: [f32; 3],
            current_voxel: [i32; 3],
            survivability: [f32; 3],
        ) -> Bound<'py, PyArray1<f32>> {
            let result = py.detach(|| {
                self.inner
                    .feature_bundle(position, current_voxel, survivability)
            });
            PyArray1::from_slice(py, &result)
        }

        fn dumps<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
            let bytes = py.detach(|| self.inner.dumps());
            PyBytes::new(py, &bytes)
        }

        #[staticmethod]
        fn loads(
            data: &[u8],
            search_radius: f32,
            voxel_size: f32,
            score_scale: f32,
        ) -> PyResult<Self> {
            let inner = LatticeIndex::loads(data, search_radius, voxel_size, score_scale)
                .map_err(PyValueError::new_err)?;
            Ok(Self { inner })
        }
    }

    #[pymodule]
    fn q2_lattice_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(nearest_signals_py, module)?)?;
        module.add_function(wrap_pyfunction!(atlas_advisory_feature_names, module)?)?;
        module.add_function(wrap_pyfunction!(dyn_feature_names, module)?)?;
        module.add_function(wrap_pyfunction!(dyn_feature_schema_sha256, module)?)?;
        module.add_function(wrap_pyfunction!(dyn_event_names, module)?)?;
        module.add_function(wrap_pyfunction!(dyn_event_schema, module)?)?;
        module.add_class::<PyLatticeIndex>()?;
        module.add_class::<PyAtlasRuntime>()?;
        module.add_class::<PyDynRuntime>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{LatticeIndex, PackedCell, nearest_signals};

    #[test]
    fn selects_each_channel_and_normalizes_direction() {
        let cells = [
            PackedCell {
                voxel: [2, 0, 0],
                raw_scores: [4.0, 0.0, 8.0, 0.0, 0.0],
                confidence: 1.0,
            },
            PackedCell {
                voxel: [-2, 0, 0],
                raw_scores: [0.0, 8.0, 0.0, 0.0, 6.0],
                confidence: 1.0,
            },
        ];
        let result = nearest_signals(&cells, [0.0; 3], 2048.0, 256.0, 8.0);
        assert!(result[0][0] > 0.0);
        assert!(result[1][0] < 0.0);
        assert!(result[2][0] > 0.0);
        assert!(result[4][0] < 0.0);
        assert_eq!(result[3], [0.0; 4]);
        assert!((result[2][3] - 1.0_f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn ranking_balances_strength_and_distance() {
        let cells = [
            PackedCell {
                voxel: [1, 0, 0],
                raw_scores: [2.0, 0.0, 0.0, 0.0, 0.0],
                confidence: 1.0,
            },
            PackedCell {
                voxel: [6, 0, 0],
                raw_scores: [8.0, 0.0, 0.0, 0.0, 0.0],
                confidence: 1.0,
            },
        ];
        let result = nearest_signals(&cells, [0.0; 3], 2048.0, 256.0, 8.0);
        assert!(result[0][0] > 0.0);
        assert!(result[0][0] < 0.3);
    }

    #[test]
    fn rejects_cells_outside_radius() {
        let cells = [PackedCell {
            voxel: [20, 0, 0],
            raw_scores: [8.0; 5],
            confidence: 1.0,
        }];
        assert_eq!(
            nearest_signals(&cells, [0.0; 3], 2048.0, 256.0, 8.0),
            [[0.0; 4]; 5]
        );
    }

    #[test]
    fn stateful_updates_replace_cells_and_fill_policy_tail() {
        let mut index = LatticeIndex::new(2048.0, 256.0, 8.0);
        index.upsert(PackedCell {
            voxel: [0, 0, 0],
            raw_scores: [4.0, 2.0, 8.0, 1.0, 3.0],
            confidence: 0.75,
        });
        index.upsert(PackedCell {
            voxel: [2, 0, 0],
            raw_scores: [8.0, 0.0, 0.0, 0.0, 0.0],
            confidence: 1.0,
        });
        assert_eq!(index.len(), 2);

        let features = index.features([0.0; 3], [0, 0, 0], [0.1, 0.2, 0.3]);
        assert!((features[0] - 0.5_f32.tanh()).abs() < 1e-6);
        assert_eq!(features[4], 0.75);
        assert!(features[5] > 0.0);
        assert_eq!(&features[21..24], &[0.1, 0.2, 0.3]);

        index.upsert(PackedCell {
            voxel: [0, 0, 0],
            raw_scores: [0.0; 5],
            confidence: 0.25,
        });
        assert_eq!(index.len(), 2);
        assert_eq!(index.features([0.0; 3], [0, 0, 0], [0.0; 3])[4], 0.25);
        assert!(index.remove([2, 0, 0]));
        assert_eq!(index.len(), 1);

        index.apply_score_event([4, 0, 0], [1.0, 2.0, 3.0, 4.0, 5.0], 1.0, false, None);
        let event_features = index.features([0.0; 3], [4, 0, 0], [0.0; 3]);
        assert!((event_features[0] - 0.125_f32.tanh()).abs() < 1e-6);
        assert!((event_features[4] - 2.0_f32.ln() / 25.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn stateful_snapshot_round_trip_is_deterministic() {
        let mut index = LatticeIndex::new(2048.0, 256.0, 8.0);
        for voxel in [[3, -2, 1], [-1, 4, 0], [0, 0, 0]] {
            index.upsert(PackedCell {
                voxel,
                raw_scores: [1.0, 2.0, 3.0, 4.0, 5.0],
                confidence: 0.5,
            });
        }
        let encoded = index.dumps();
        let restored = LatticeIndex::loads(&encoded, 2048.0, 256.0, 8.0).unwrap();
        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.dumps(), encoded);
        assert_eq!(
            restored.nearest_signals([100.0, 200.0, 0.0]),
            index.nearest_signals([100.0, 200.0, 0.0])
        );
        assert!(LatticeIndex::loads(b"bad", 1.0, 1.0, 1.0).is_err());
    }
}
