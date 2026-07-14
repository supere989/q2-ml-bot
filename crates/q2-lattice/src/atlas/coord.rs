use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::{AtlasError, AtlasResult};

pub const ATLAS_CELL_SIZES: [i64; 4] = [4, 16, 64, 256];
const ORIGIN_SNAP: i64 = 256;
const L0_CHUNK_SIDE_I32: i32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AtlasLevel {
    L0 = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
}

impl AtlasLevel {
    pub const ALL: [Self; 4] = [Self::L0, Self::L1, Self::L2, Self::L3];

    pub const fn cell_size(self) -> i64 {
        ATLAS_CELL_SIZES[self as usize]
    }
}

impl TryFrom<u8> for AtlasLevel {
    type Error = AtlasError;

    fn try_from(value: u8) -> AtlasResult<Self> {
        match value {
            0 => Ok(Self::L0),
            1 => Ok(Self::L1),
            2 => Ok(Self::L2),
            3 => Ok(Self::L3),
            _ => Err(AtlasError::InvalidFormat(format!("unknown level {value}"))),
        }
    }
}

/// Signed grid coordinate. Ordering is the canonical Atlas order `(iz,iy,ix)`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct GridIndex {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl GridIndex {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub const fn xyz(self) -> [i32; 3] {
        [self.x, self.y, self.z]
    }

    /// Mathematical floor division by four, including negative nonmultiples.
    pub fn parent(self) -> Self {
        Self::new(
            self.x.div_euclid(4),
            self.y.div_euclid(4),
            self.z.div_euclid(4),
        )
    }

    pub fn child_min(self) -> AtlasResult<Self> {
        Ok(Self::new(
            self.x
                .checked_mul(4)
                .ok_or_else(|| AtlasError::Coordinate("child x exceeds i32".to_owned()))?,
            self.y
                .checked_mul(4)
                .ok_or_else(|| AtlasError::Coordinate("child y exceeds i32".to_owned()))?,
            self.z
                .checked_mul(4)
                .ok_or_else(|| AtlasError::Coordinate("child z exceeds i32".to_owned()))?,
        ))
    }

    pub fn child(self, x: u8, y: u8, z: u8) -> AtlasResult<Self> {
        if x >= 4 || y >= 4 || z >= 4 {
            return Err(AtlasError::Coordinate(
                "child offsets must be in 0..4".to_owned(),
            ));
        }
        let minimum = self.child_min()?;
        Ok(Self::new(
            minimum.x + i32::from(x),
            minimum.y + i32::from(y),
            minimum.z + i32::from(z),
        ))
    }
}

impl Ord for GridIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.z, self.y, self.x).cmp(&(other.z, other.y, other.x))
    }
}

impl PartialOrd for GridIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AtlasOrigin(pub [i64; 3]);

impl AtlasOrigin {
    pub fn snapped(integer_model0_mins: [i64; 3]) -> AtlasResult<Self> {
        let mut snapped = [0_i64; 3];
        for (destination, value) in snapped.iter_mut().zip(integer_model0_mins) {
            *destination = value
                .div_euclid(ORIGIN_SNAP)
                .checked_mul(ORIGIN_SNAP)
                .ok_or_else(|| AtlasError::Coordinate("snapped origin exceeds i64".to_owned()))?;
        }
        Ok(Self(snapped))
    }

    pub fn index_integer(self, world: [i64; 3], level: AtlasLevel) -> AtlasResult<GridIndex> {
        let size = level.cell_size();
        let mut index = [0_i32; 3];
        for axis in 0..3 {
            let relative = world[axis].checked_sub(self.0[axis]).ok_or_else(|| {
                AtlasError::Coordinate("world-origin subtraction overflow".to_owned())
            })?;
            index[axis] = i32::try_from(relative.div_euclid(size)).map_err(|_| {
                AtlasError::Coordinate(format!(
                    "level {} index on axis {axis} exceeds i32",
                    level as u8
                ))
            })?;
        }
        Ok(GridIndex::new(index[0], index[1], index[2]))
    }

    pub fn index(self, world: [f64; 3], level: AtlasLevel) -> AtlasResult<GridIndex> {
        let size = level.cell_size() as f64;
        let mut index = [0_i32; 3];
        for axis in 0..3 {
            if !world[axis].is_finite() {
                return Err(AtlasError::Coordinate(format!(
                    "non-finite world coordinate on axis {axis}"
                )));
            }
            let value = ((world[axis] - self.0[axis] as f64) / size).floor();
            if value < i32::MIN as f64 || value > i32::MAX as f64 {
                return Err(AtlasError::Coordinate(format!(
                    "level {} index on axis {axis} exceeds i32",
                    level as u8
                )));
            }
            index[axis] = value as i32;
        }
        Ok(GridIndex::new(index[0], index[1], index[2]))
    }

    pub fn center(self, index: GridIndex, level: AtlasLevel) -> [f64; 3] {
        let size = level.cell_size() as f64;
        [
            self.0[0] as f64 + (index.x as f64 + 0.5) * size,
            self.0[1] as f64 + (index.y as f64 + 0.5) * size,
            self.0[2] as f64 + (index.z as f64 + 0.5) * size,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct L0Address {
    pub chunk: GridIndex,
    pub local: [u8; 3],
    pub linear: u16,
}

impl L0Address {
    pub fn from_l0_index(index: GridIndex) -> Self {
        let chunk = GridIndex::new(
            index.x.div_euclid(L0_CHUNK_SIDE_I32),
            index.y.div_euclid(L0_CHUNK_SIDE_I32),
            index.z.div_euclid(L0_CHUNK_SIDE_I32),
        );
        let local = [
            index.x.rem_euclid(L0_CHUNK_SIDE_I32) as u8,
            index.y.rem_euclid(L0_CHUNK_SIDE_I32) as u8,
            index.z.rem_euclid(L0_CHUNK_SIDE_I32) as u8,
        ];
        let linear = u16::from(local[0]) + u16::from(local[1]) * 16 + u16::from(local[2]) * 256;
        Self {
            chunk,
            local,
            linear,
        }
    }
}
