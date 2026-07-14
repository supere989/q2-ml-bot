//! Hot-path kernels for q2-ml-bot's per-environment Vector Lattice.
//!
//! The policy contract remains a 24-float observation tail. This crate starts
//! with the measured remaining kernel: choose the strongest nearby signal for
//! five channels in one pass over packed cells. Python remains authoritative
//! for deposits, rewards, persistence, and orchestration until parity and
//! end-to-end SPS gates justify moving more state here.

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
    use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

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
        module.add_class::<PyLatticeIndex>()?;
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
