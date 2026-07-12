//! Hot-path kernels for q2-ml-bot's per-environment Vector Lattice.
//!
//! The policy contract remains a 24-float observation tail. This crate starts
//! with the measured remaining kernel: choose the strongest nearby signal for
//! five channels in one pass over packed cells. Python remains authoritative
//! for deposits, rewards, persistence, and orchestration until parity and
//! end-to-end SPS gates justify moving more state here.

const CHANNELS: usize = 5;
const OUTPUTS: usize = 4;
pub const PACKED_CELL_WIDTH: usize = 9;

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
        for channel in 0..CHANNELS {
            let raw = cell.raw_scores[channel];
            if raw <= 0.0 {
                continue;
            }
            let score = (raw / scale).tanh() * cell.confidence;
            let rank = score / distance_scale;
            if rank > best[channel].rank {
                best[channel] = Candidate {
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

#[cfg(feature = "python")]
mod python {
    use super::{PACKED_CELL_WIDTH, PackedCell, nearest_signals};
    use numpy::{PyArray2, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction(name = "nearest_signals")]
    fn nearest_signals_py<'py>(
        py: Python<'py>,
        cells: PyReadonlyArray2<'py, f32>,
        position: [f32; 3],
        search_radius: f32,
        voxel_size: f32,
        score_scale: f32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let view = cells.as_array();
        if view.shape().len() != 2 || view.shape()[1] != PACKED_CELL_WIDTH {
            return Err(PyValueError::new_err(format!(
                "cells must have shape (N, {PACKED_CELL_WIDTH})"
            )));
        }
        let cell_values = cells
            .as_slice()
            .map_err(|_| PyValueError::new_err("cells must be a C-contiguous float32 array"))?;
        let packed: Vec<PackedCell> = cell_values
            .chunks_exact(PACKED_CELL_WIDTH)
            .map(|row| PackedCell::from_row(row).expect("chunk width was validated"))
            .collect();
        let result = nearest_signals(&packed, position, search_radius, voxel_size, score_scale);
        let rows: Vec<Vec<f32>> = result.into_iter().map(Vec::from).collect();
        PyArray2::from_vec2(py, &rows).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pymodule]
    fn q2_lattice_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(nearest_signals_py, module)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PackedCell, nearest_signals};

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
}
