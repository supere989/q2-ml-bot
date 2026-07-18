use std::collections::BTreeMap;

use super::{AtlasError, AtlasLimits, AtlasResult, GridIndex};

pub const L0_CHUNK_SIDE: usize = 16;
pub const L0_CELLS_PER_CHUNK: usize = L0_CHUNK_SIDE * L0_CHUNK_SIDE * L0_CHUNK_SIDE;
pub(crate) const BITPLANE_WORDS: usize = L0_CELLS_PER_CHUNK / 64;
pub(crate) const BITPLANE_BYTES: usize = BITPLANE_WORDS * 8;

/// Binary L0 channels. Discriminants are part of Atlas schema v1.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum L0BitPlane {
    Solid = 0,
    Window = 1,
    PlayerClip = 2,
    MonsterClipDiagnostic = 3,
    Water = 4,
    Slime = 5,
    Lava = 6,
    Mist = 7,
    Ladder = 8,
    Hurt = 9,
    PushOrGravity = 10,
    TeleportTrigger = 11,
    MoverReferenceSolid = 12,
    MoverSweptEnvelope = 13,
    AreaPortal = 14,
    Sky = 15,
    Slick = 16,
    Warp = 17,
    NoDrawDiagnostic = 18,
    HookableSurface = 19,
    StandingForbiddenOrigin = 20,
    CrouchedForbiddenOrigin = 21,
    Unknown = 22,
    SpawnColumn = 23,
    HookCorridor = 24,
}

impl L0BitPlane {
    pub const COUNT: usize = 25;

    pub(crate) fn from_u8(value: u8) -> AtlasResult<Self> {
        match value {
            0 => Ok(Self::Solid),
            1 => Ok(Self::Window),
            2 => Ok(Self::PlayerClip),
            3 => Ok(Self::MonsterClipDiagnostic),
            4 => Ok(Self::Water),
            5 => Ok(Self::Slime),
            6 => Ok(Self::Lava),
            7 => Ok(Self::Mist),
            8 => Ok(Self::Ladder),
            9 => Ok(Self::Hurt),
            10 => Ok(Self::PushOrGravity),
            11 => Ok(Self::TeleportTrigger),
            12 => Ok(Self::MoverReferenceSolid),
            13 => Ok(Self::MoverSweptEnvelope),
            14 => Ok(Self::AreaPortal),
            15 => Ok(Self::Sky),
            16 => Ok(Self::Slick),
            17 => Ok(Self::Warp),
            18 => Ok(Self::NoDrawDiagnostic),
            19 => Ok(Self::HookableSurface),
            20 => Ok(Self::StandingForbiddenOrigin),
            21 => Ok(Self::CrouchedForbiddenOrigin),
            22 => Ok(Self::Unknown),
            23 => Ok(Self::SpawnColumn),
            24 => Ok(Self::HookCorridor),
            _ => Err(AtlasError::InvalidFormat(format!(
                "unknown L0 bitplane {value}"
            ))),
        }
    }
}

/// Compact integer L0 channels. Missing planes are canonically all zero.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum L0ScalarPlane {
    CurrentDirection = 0,
    HazardSeverity = 1,
    Confidence = 2,
}

impl L0ScalarPlane {
    pub const COUNT: usize = 3;

    pub(crate) fn from_u8(value: u8) -> AtlasResult<Self> {
        match value {
            0 => Ok(Self::CurrentDirection),
            1 => Ok(Self::HazardSeverity),
            2 => Ok(Self::Confidence),
            _ => Err(AtlasError::InvalidFormat(format!(
                "unknown L0 scalar plane {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct L0Chunk {
    key: GridIndex,
    bitplanes: BTreeMap<L0BitPlane, Box<[u64; BITPLANE_WORDS]>>,
    scalar_planes: BTreeMap<L0ScalarPlane, Box<[u8; L0_CELLS_PER_CHUNK]>>,
}

impl L0Chunk {
    pub fn new(key: GridIndex) -> Self {
        Self {
            key,
            bitplanes: BTreeMap::new(),
            scalar_planes: BTreeMap::new(),
        }
    }

    pub const fn key(&self) -> GridIndex {
        self.key
    }

    pub fn is_empty(&self) -> bool {
        self.bitplanes.is_empty() && self.scalar_planes.is_empty()
    }

    pub fn bitplane_count(&self) -> usize {
        self.bitplanes.len()
    }

    pub fn scalar_plane_count(&self) -> usize {
        self.scalar_planes.len()
    }

    pub fn set_bit(&mut self, plane: L0BitPlane, linear: usize, value: bool) -> AtlasResult<()> {
        if linear >= L0_CELLS_PER_CHUNK {
            return Err(AtlasError::Coordinate(format!(
                "L0 linear cell {linear} is outside a 16^3 chunk"
            )));
        }
        let word_index = linear / 64;
        let mask = 1_u64 << (linear % 64);
        if value {
            let words = self
                .bitplanes
                .entry(plane)
                .or_insert_with(|| Box::new([0; BITPLANE_WORDS]));
            words[word_index] |= mask;
        } else if let Some(words) = self.bitplanes.get_mut(&plane) {
            words[word_index] &= !mask;
            if words.iter().all(|word| *word == 0) {
                self.bitplanes.remove(&plane);
            }
        }
        Ok(())
    }

    pub fn bit(&self, plane: L0BitPlane, linear: usize) -> AtlasResult<bool> {
        if linear >= L0_CELLS_PER_CHUNK {
            return Err(AtlasError::Coordinate(format!(
                "L0 linear cell {linear} is outside a 16^3 chunk"
            )));
        }
        Ok(self
            .bitplanes
            .get(&plane)
            .is_some_and(|words| words[linear / 64] & (1_u64 << (linear % 64)) != 0))
    }

    pub fn set_scalar(
        &mut self,
        plane: L0ScalarPlane,
        linear: usize,
        value: u8,
    ) -> AtlasResult<()> {
        if linear >= L0_CELLS_PER_CHUNK {
            return Err(AtlasError::Coordinate(format!(
                "L0 linear cell {linear} is outside a 16^3 chunk"
            )));
        }
        if value != 0 {
            let values = self
                .scalar_planes
                .entry(plane)
                .or_insert_with(|| Box::new([0; L0_CELLS_PER_CHUNK]));
            values[linear] = value;
        } else if let Some(values) = self.scalar_planes.get_mut(&plane) {
            values[linear] = 0;
            if values.iter().all(|entry| *entry == 0) {
                self.scalar_planes.remove(&plane);
            }
        }
        Ok(())
    }

    pub fn scalar(&self, plane: L0ScalarPlane, linear: usize) -> AtlasResult<u8> {
        if linear >= L0_CELLS_PER_CHUNK {
            return Err(AtlasError::Coordinate(format!(
                "L0 linear cell {linear} is outside a 16^3 chunk"
            )));
        }
        Ok(self
            .scalar_planes
            .get(&plane)
            .map_or(0, |values| values[linear]))
    }

    /// Canonical payload contribution, excluding the containing section header.
    pub fn encoded_len(&self) -> usize {
        12 + 8
            + 1
            + self.bitplanes.len() * BITPLANE_BYTES
            + self.scalar_planes.len() * L0_CELLS_PER_CHUNK
    }

    pub fn payload_bytes(&self) -> usize {
        self.bitplanes.len() * BITPLANE_BYTES + self.scalar_planes.len() * L0_CELLS_PER_CHUNK
    }

    pub(crate) fn bitplanes(&self) -> &BTreeMap<L0BitPlane, Box<[u64; BITPLANE_WORDS]>> {
        &self.bitplanes
    }

    pub(crate) fn scalar_planes(&self) -> &BTreeMap<L0ScalarPlane, Box<[u8; L0_CELLS_PER_CHUNK]>> {
        &self.scalar_planes
    }

    pub(crate) fn from_planes(
        key: GridIndex,
        bitplanes: BTreeMap<L0BitPlane, Box<[u64; BITPLANE_WORDS]>>,
        scalar_planes: BTreeMap<L0ScalarPlane, Box<[u8; L0_CELLS_PER_CHUNK]>>,
    ) -> AtlasResult<Self> {
        if bitplanes
            .values()
            .any(|plane| plane.iter().all(|word| *word == 0))
        {
            return Err(AtlasError::InvalidFormat(
                "canonical L0 contains an empty bitplane".to_owned(),
            ));
        }
        if scalar_planes
            .values()
            .any(|plane| plane.iter().all(|value| *value == 0))
        {
            return Err(AtlasError::InvalidFormat(
                "canonical L0 contains an empty scalar plane".to_owned(),
            ));
        }
        let chunk = Self {
            key,
            bitplanes,
            scalar_planes,
        };
        if chunk.is_empty() {
            return Err(AtlasError::InvalidFormat(
                "canonical L0 contains an empty chunk".to_owned(),
            ));
        }
        Ok(chunk)
    }
}

/// Sparse-only L0 storage. There is intentionally no AABB/dense constructor.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SparseL0 {
    chunks: BTreeMap<GridIndex, L0Chunk>,
    encoded_bytes: usize,
}

impl SparseL0 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }

    /// Conservative heap estimate used by the runtime admission guard. Plane
    /// payloads are exact; map-node overhead is intentionally rounded up.
    pub fn resident_bytes_estimate(&self) -> usize {
        const BTREE_ENTRY_OVERHEAD: usize = 64;
        self.chunks
            .values()
            .fold(std::mem::size_of::<Self>(), |total, chunk| {
                total
                    + std::mem::size_of::<GridIndex>()
                    + std::mem::size_of::<L0Chunk>()
                    + BTREE_ENTRY_OVERHEAD
                    + chunk.payload_bytes()
                    + (chunk.bitplane_count() + chunk.scalar_plane_count()) * BTREE_ENTRY_OVERHEAD
            })
    }

    pub fn chunks(&self) -> impl ExactSizeIterator<Item = &L0Chunk> {
        self.chunks.values()
    }

    pub fn get(&self, key: GridIndex) -> Option<&L0Chunk> {
        self.chunks.get(&key)
    }

    pub fn insert(&mut self, chunk: L0Chunk, limits: &AtlasLimits) -> AtlasResult<()> {
        if chunk.is_empty() {
            return Err(AtlasError::InvalidFormat(
                "empty L0 chunks are not materialized".to_owned(),
            ));
        }
        let replaced = self
            .chunks
            .get(&chunk.key())
            .map_or(0, L0Chunk::encoded_len);
        let prospective_count = self.chunks.len() + usize::from(replaced == 0);
        if prospective_count > limits.max_l0_chunks {
            return Err(AtlasError::LimitExceeded(format!(
                "L0 chunk count {prospective_count} > {}",
                limits.max_l0_chunks
            )));
        }
        let prospective_bytes = self
            .encoded_bytes
            .checked_sub(replaced)
            .and_then(|value| value.checked_add(chunk.encoded_len()))
            .ok_or_else(|| AtlasError::LimitExceeded("L0 byte count overflow".to_owned()))?;
        if prospective_bytes > limits.max_l0_decompressed_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "L0 bytes {prospective_bytes} > {}",
                limits.max_l0_decompressed_bytes
            )));
        }
        self.encoded_bytes = prospective_bytes;
        self.chunks.insert(chunk.key(), chunk);
        Ok(())
    }
}
