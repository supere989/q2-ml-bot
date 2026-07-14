use std::collections::BTreeMap;
use std::io::{Cursor, Read};

use sha2::{Digest, Sha256};

use super::graph::{EdgeRecord, EdgeType, L1Node, Stance};
use super::l0::{BITPLANE_WORDS, L0_CELLS_PER_CHUNK, L0BitPlane, L0Chunk, L0ScalarPlane, SparseL0};
use super::{ATLAS_CELL_SIZES, AtlasError, AtlasOrigin, AtlasResult, GridIndex, L1Graph};

pub const ATLAS_MAGIC: &[u8; 8] = b"Q2ATL001";
pub const ENVELOPE_MAGIC: &[u8; 8] = b"Q2AZS001";
pub const ATLAS_SCHEMA_VERSION: u16 = 1;
const BYTE_ORDER_LITTLE: u16 = 0x454c;
const HEADER_BYTES: u32 = 136;
const ENVELOPE_HEADER_BYTES: usize = 64;
const ZSTD_LEVEL: i8 = 3;
const NODE_BYTES: usize = 40;
const AGGREGATE_BYTES: usize = 28;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtlasLimits {
    pub max_l0_chunks: usize,
    pub max_l0_decompressed_bytes: usize,
    pub max_atlas_decompressed_bytes: usize,
    pub max_atlas_resident_bytes: usize,
    pub max_compressed_payload_bytes: usize,
    pub max_build_rss_bytes: u64,
    pub max_manifest_bytes: usize,
    pub max_l1_nodes: usize,
    pub max_l1_edges: usize,
    pub max_l2_cells: usize,
    pub max_l3_cells: usize,
}

impl Default for AtlasLimits {
    fn default() -> Self {
        Self {
            max_l0_chunks: 1_200,
            max_l0_decompressed_bytes: 16 * 1024 * 1024,
            max_atlas_decompressed_bytes: 32 * 1024 * 1024,
            max_atlas_resident_bytes: 32 * 1024 * 1024,
            max_compressed_payload_bytes: 32 * 1024 * 1024,
            max_build_rss_bytes: 512 * 1024 * 1024,
            max_manifest_bytes: 1024 * 1024,
            max_l1_nodes: 500_000,
            max_l1_edges: 1_500_000,
            max_l2_cells: 100_000,
            max_l3_cells: 25_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtlasAggregateCell {
    pub index: GridIndex,
    pub contents_flags: u32,
    pub hazard_types: u16,
    pub hazard_severity: u8,
    pub standing_passable: bool,
    pub crouched_passable: bool,
    pub clearance: u16,
    pub cost_to_safety: u32,
    pub confidence: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtlasArtifact {
    pub origin: AtlasOrigin,
    pub l0: SparseL0,
    pub l1: L1Graph,
    pub l2: Vec<AtlasAggregateCell>,
    pub l3: Vec<AtlasAggregateCell>,
}

impl AtlasArtifact {
    pub fn empty(origin: AtlasOrigin) -> Self {
        Self {
            origin,
            l0: SparseL0::new(),
            l1: L1Graph::empty(),
            l2: Vec::new(),
            l3: Vec::new(),
        }
    }

    pub fn validate(&self, limits: &AtlasLimits) -> AtlasResult<()> {
        if self.origin.0.iter().any(|value| value.rem_euclid(256) != 0) {
            return Err(AtlasError::InvalidFormat(
                "Atlas origin is not snapped to 256 units".to_owned(),
            ));
        }
        if self.l0.len() > limits.max_l0_chunks
            || self.l0.encoded_bytes() > limits.max_l0_decompressed_bytes
        {
            return Err(AtlasError::LimitExceeded(
                "sparse L0 exceeds chunk/byte budget".to_owned(),
            ));
        }
        self.l1.validate(limits)?;
        validate_aggregate_cells(&self.l2, limits.max_l2_cells, "L2")?;
        validate_aggregate_cells(&self.l3, limits.max_l3_cells, "L3")?;
        let resident = self.resident_bytes_estimate();
        if resident > limits.max_atlas_resident_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "Atlas resident estimate {resident} > {}",
                limits.max_atlas_resident_bytes
            )));
        }
        Ok(())
    }

    pub fn resident_bytes_estimate(&self) -> usize {
        self.l0.resident_bytes_estimate()
            + std::mem::size_of::<Self>()
            + self.l1.resident_bytes_estimate()
            + (self.l2.capacity() + self.l3.capacity()) * std::mem::size_of::<AtlasAggregateCell>()
    }

    /// Canonical uncompressed little-endian bytes. This payload, not the zstd
    /// envelope, owns the authoritative SHA-256 identity.
    pub fn encode_uncompressed(&self, limits: &AtlasLimits) -> AtlasResult<Vec<u8>> {
        self.validate(limits)?;
        let l0 = encode_l0(&self.l0);
        let nodes = encode_nodes(self.l1.nodes());
        let graph = encode_graph(&self.l1);
        let l2 = encode_aggregate_cells(&self.l2);
        let l3 = encode_aggregate_cells(&self.l3);
        if l0.len() > limits.max_l0_decompressed_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "L0 section bytes {} > {}",
                l0.len(),
                limits.max_l0_decompressed_bytes
            )));
        }
        let total = [l0.len(), nodes.len(), graph.len(), l2.len(), l3.len()]
            .into_iter()
            .try_fold(HEADER_BYTES as usize, |total, length| {
                total.checked_add(length).ok_or_else(|| {
                    AtlasError::LimitExceeded("Atlas encoded byte count overflow".to_owned())
                })
            })?;
        if total > limits.max_atlas_decompressed_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "Atlas bytes {total} > {}",
                limits.max_atlas_decompressed_bytes
            )));
        }
        let mut output = Vec::with_capacity(total);
        output.extend_from_slice(ATLAS_MAGIC);
        push_u16(&mut output, ATLAS_SCHEMA_VERSION);
        push_u16(&mut output, BYTE_ORDER_LITTLE);
        push_u32(&mut output, HEADER_BYTES);
        for value in self.origin.0 {
            push_i64(&mut output, value);
        }
        for size in ATLAS_CELL_SIZES {
            push_u32(&mut output, size as u32);
        }
        for count in [
            self.l0.len(),
            self.l1.nodes().len(),
            self.l1.edges().len(),
            self.l2.len(),
            self.l3.len(),
        ] {
            push_u64(&mut output, count as u64);
        }
        for length in [l0.len(), nodes.len(), graph.len(), l2.len(), l3.len()] {
            push_u64(&mut output, length as u64);
        }
        debug_assert_eq!(output.len(), HEADER_BYTES as usize);
        output.extend_from_slice(&l0);
        output.extend_from_slice(&nodes);
        output.extend_from_slice(&graph);
        output.extend_from_slice(&l2);
        output.extend_from_slice(&l3);
        Ok(output)
    }

    pub fn encode_zstd(&self, limits: &AtlasLimits) -> AtlasResult<Vec<u8>> {
        encode_zstd_envelope(&self.encode_uncompressed(limits)?, limits)
    }

    pub fn decode_uncompressed(bytes: &[u8], limits: &AtlasLimits) -> AtlasResult<Self> {
        if bytes.len() > limits.max_atlas_decompressed_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "Atlas bytes {} > {}",
                bytes.len(),
                limits.max_atlas_decompressed_bytes
            )));
        }
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != ATLAS_MAGIC {
            return Err(AtlasError::InvalidFormat("invalid Atlas magic".to_owned()));
        }
        let schema = reader.u16()?;
        if schema != ATLAS_SCHEMA_VERSION {
            return Err(AtlasError::MixedSchema {
                expected: ATLAS_SCHEMA_VERSION,
                found: schema,
            });
        }
        if reader.u16()? != BYTE_ORDER_LITTLE {
            return Err(AtlasError::InvalidFormat(
                "Atlas byte order is not little-endian".to_owned(),
            ));
        }
        if reader.u32()? != HEADER_BYTES {
            return Err(AtlasError::InvalidFormat(
                "unexpected Atlas header size".to_owned(),
            ));
        }
        let origin = AtlasOrigin([reader.i64()?, reader.i64()?, reader.i64()?]);
        for expected in ATLAS_CELL_SIZES {
            if reader.u32()? != expected as u32 {
                return Err(AtlasError::InvalidFormat(
                    "mixed Atlas cell-size schema".to_owned(),
                ));
            }
        }
        let counts = [
            reader.count(limits.max_l0_chunks, "L0 chunks")?,
            reader.count(limits.max_l1_nodes, "L1 nodes")?,
            reader.count(limits.max_l1_edges, "L1 edges")?,
            reader.count(limits.max_l2_cells, "L2 cells")?,
            reader.count(limits.max_l3_cells, "L3 cells")?,
        ];
        let mut lengths = [0_usize; 5];
        for length in &mut lengths {
            *length = usize::try_from(reader.u64()?).map_err(|_| {
                AtlasError::LimitExceeded("section length exceeds usize".to_owned())
            })?;
        }
        if lengths[0] > limits.max_l0_decompressed_bytes {
            return Err(AtlasError::LimitExceeded(
                "L0 section exceeds byte budget".to_owned(),
            ));
        }
        let sections_total = lengths
            .iter()
            .try_fold(HEADER_BYTES as usize, |sum, length| {
                sum.checked_add(*length).ok_or_else(|| {
                    AtlasError::LimitExceeded("Atlas section lengths overflow".to_owned())
                })
            })?;
        if sections_total != bytes.len() {
            return Err(AtlasError::InvalidFormat(format!(
                "Atlas section lengths total {sections_total}, payload is {}",
                bytes.len()
            )));
        }
        let l0 = decode_l0(reader.subreader(lengths[0])?, counts[0], limits)?;
        let nodes = decode_nodes(reader.subreader(lengths[1])?, counts[1])?;
        let (offsets, edges) = decode_graph(reader.subreader(lengths[2])?, counts[1], counts[2])?;
        let l1 = L1Graph::from_canonical_parts(nodes, offsets, edges, limits)?;
        let l2 = decode_aggregate_cells(reader.subreader(lengths[3])?, counts[3], "L2")?;
        let l3 = decode_aggregate_cells(reader.subreader(lengths[4])?, counts[4], "L3")?;
        if !reader.is_empty() {
            return Err(AtlasError::InvalidFormat("trailing Atlas bytes".to_owned()));
        }
        let artifact = Self {
            origin,
            l0,
            l1,
            l2,
            l3,
        };
        artifact.validate(limits)?;
        Ok(artifact)
    }

    pub fn decode_zstd(envelope: &[u8], limits: &AtlasLimits) -> AtlasResult<Self> {
        let payload = decode_zstd_envelope(envelope, limits)?;
        Self::decode_uncompressed(&payload, limits)
    }
}

pub fn encode_zstd_envelope(payload: &[u8], limits: &AtlasLimits) -> AtlasResult<Vec<u8>> {
    if payload.len() > limits.max_atlas_decompressed_bytes {
        return Err(AtlasError::LimitExceeded(format!(
            "uncompressed envelope payload {} > {}",
            payload.len(),
            limits.max_atlas_decompressed_bytes
        )));
    }
    let compressed = zstd::stream::encode_all(Cursor::new(payload), i32::from(ZSTD_LEVEL))?;
    if compressed.len() > limits.max_compressed_payload_bytes {
        return Err(AtlasError::LimitExceeded(format!(
            "compressed envelope payload {} > {}",
            compressed.len(),
            limits.max_compressed_payload_bytes
        )));
    }
    let digest: [u8; 32] = Sha256::digest(payload).into();
    let mut output = Vec::with_capacity(ENVELOPE_HEADER_BYTES + compressed.len());
    output.extend_from_slice(ENVELOPE_MAGIC);
    push_u16(&mut output, ATLAS_SCHEMA_VERSION);
    output.push(1); // zstd, no dictionary
    output.push(ZSTD_LEVEL as u8);
    push_u32(&mut output, 0);
    push_u64(&mut output, payload.len() as u64);
    push_u64(&mut output, compressed.len() as u64);
    output.extend_from_slice(&digest);
    debug_assert_eq!(output.len(), ENVELOPE_HEADER_BYTES);
    output.extend_from_slice(&compressed);
    Ok(output)
}

pub fn decode_zstd_envelope(envelope: &[u8], limits: &AtlasLimits) -> AtlasResult<Vec<u8>> {
    let mut reader = Reader::new(envelope);
    if reader.take(8)? != ENVELOPE_MAGIC {
        return Err(AtlasError::InvalidFormat(
            "invalid Atlas envelope magic".to_owned(),
        ));
    }
    let schema = reader.u16()?;
    if schema != ATLAS_SCHEMA_VERSION {
        return Err(AtlasError::MixedSchema {
            expected: ATLAS_SCHEMA_VERSION,
            found: schema,
        });
    }
    if reader.u8()? != 1 || reader.u8()? != ZSTD_LEVEL as u8 || reader.u32()? != 0 {
        return Err(AtlasError::InvalidFormat(
            "unsupported Atlas compression envelope".to_owned(),
        ));
    }
    let uncompressed_len = reader.count(
        limits.max_atlas_decompressed_bytes,
        "uncompressed envelope bytes",
    )?;
    let compressed_len = reader.count(
        limits.max_compressed_payload_bytes,
        "compressed envelope bytes",
    )?;
    let expected_digest: [u8; 32] = reader.take(32)?.try_into().unwrap();
    if reader.remaining() != compressed_len {
        return Err(AtlasError::InvalidFormat(
            "compressed envelope length mismatch".to_owned(),
        ));
    }
    let compressed = reader.take(compressed_len)?;
    let mut decoder = zstd::stream::read::Decoder::new(Cursor::new(compressed))?;
    let window_log = usize::BITS - limits.max_atlas_decompressed_bytes.max(1).leading_zeros();
    decoder.window_log_max(window_log)?;
    let read_limit = (uncompressed_len as u64)
        .checked_add(1)
        .ok_or_else(|| AtlasError::LimitExceeded("uncompressed read limit overflow".to_owned()))?;
    let mut bounded = decoder.take(read_limit);
    let mut payload = Vec::with_capacity(uncompressed_len);
    bounded.read_to_end(&mut payload)?;
    if payload.len() != uncompressed_len {
        return Err(AtlasError::InvalidFormat(format!(
            "uncompressed envelope length mismatch: expected {uncompressed_len}, got {}",
            payload.len()
        )));
    }
    let actual_digest: [u8; 32] = Sha256::digest(&payload).into();
    if actual_digest != expected_digest {
        return Err(AtlasError::DigestMismatch);
    }
    Ok(payload)
}

fn validate_aggregate_cells(
    cells: &[AtlasAggregateCell],
    maximum: usize,
    level: &str,
) -> AtlasResult<()> {
    if cells.len() > maximum {
        return Err(AtlasError::LimitExceeded(format!(
            "{level} cell count {} > {maximum}",
            cells.len()
        )));
    }
    if cells.windows(2).any(|pair| pair[0].index >= pair[1].index) {
        return Err(AtlasError::InvalidFormat(format!(
            "{level} cells are not strictly ordered (iz,iy,ix)"
        )));
    }
    Ok(())
}

fn encode_l0(l0: &SparseL0) -> Vec<u8> {
    let mut output = Vec::with_capacity(l0.encoded_bytes());
    for chunk in l0.chunks() {
        push_index(&mut output, chunk.key());
        let mask = chunk
            .bitplanes()
            .keys()
            .fold(0_u64, |mask, plane| mask | (1_u64 << (*plane as u8)));
        push_u64(&mut output, mask);
        let scalar_mask = chunk
            .scalar_planes()
            .keys()
            .fold(0_u8, |mask, plane| mask | (1_u8 << (*plane as u8)));
        output.push(scalar_mask);
        for words in chunk.bitplanes().values() {
            for word in words.iter() {
                push_u64(&mut output, *word);
            }
        }
        for values in chunk.scalar_planes().values() {
            output.extend_from_slice(values.as_slice());
        }
    }
    output
}

fn decode_l0(mut reader: Reader<'_>, count: usize, limits: &AtlasLimits) -> AtlasResult<SparseL0> {
    let mut l0 = SparseL0::new();
    let mut previous = None;
    for _ in 0..count {
        let key = reader.index()?;
        if previous.is_some_and(|value| key <= value) {
            return Err(AtlasError::InvalidFormat(
                "L0 chunks are not strictly ordered (iz,iy,ix)".to_owned(),
            ));
        }
        previous = Some(key);
        let mask = reader.u64()?;
        if mask >> L0BitPlane::COUNT != 0 {
            return Err(AtlasError::InvalidFormat(
                "L0 chunk has unknown bitplanes".to_owned(),
            ));
        }
        let scalar_mask = reader.u8()?;
        if scalar_mask >> L0ScalarPlane::COUNT != 0 {
            return Err(AtlasError::InvalidFormat(
                "L0 chunk has unknown scalar planes".to_owned(),
            ));
        }
        let mut bitplanes = BTreeMap::new();
        for plane_index in 0..L0BitPlane::COUNT as u8 {
            if mask & (1_u64 << plane_index) == 0 {
                continue;
            }
            let mut words = Box::new([0_u64; BITPLANE_WORDS]);
            for word in words.iter_mut() {
                *word = reader.u64()?;
            }
            bitplanes.insert(L0BitPlane::from_u8(plane_index)?, words);
        }
        let mut scalar_planes = BTreeMap::new();
        for plane_index in 0..L0ScalarPlane::COUNT as u8 {
            if scalar_mask & (1_u8 << plane_index) == 0 {
                continue;
            }
            let values: Box<[u8; L0_CELLS_PER_CHUNK]> = reader
                .take(L0_CELLS_PER_CHUNK)?
                .try_into()
                .map(Box::new)
                .map_err(|_| AtlasError::InvalidFormat("bad scalar plane".to_owned()))?;
            scalar_planes.insert(L0ScalarPlane::from_u8(plane_index)?, values);
        }
        l0.insert(L0Chunk::from_planes(key, bitplanes, scalar_planes)?, limits)?;
    }
    reader.finish("L0")?;
    Ok(l0)
}

fn encode_nodes(nodes: &[L1Node]) -> Vec<u8> {
    let mut output = Vec::with_capacity(nodes.len() * NODE_BYTES);
    for node in nodes {
        push_index(&mut output, node.index);
        push_u16(&mut output, node.flags);
        output.push(node.floor_normal_class);
        output.push(node.hazard_severity);
        push_u16(&mut output, node.clearance_height);
        push_u16(&mut output, node.hazard_types);
        push_i32(&mut output, node.hazard_clearance);
        push_u32(&mut output, node.cost_to_safety);
        push_u32(&mut output, node.region_id);
        push_u16(&mut output, node.confidence);
        push_u16(&mut output, node.evidence);
        push_u32(&mut output, node.contents_flags);
    }
    output
}

fn decode_nodes(mut reader: Reader<'_>, count: usize) -> AtlasResult<Vec<L1Node>> {
    if reader.remaining()
        != count
            .checked_mul(NODE_BYTES)
            .ok_or_else(|| AtlasError::LimitExceeded("L1 node byte count overflow".to_owned()))?
    {
        return Err(AtlasError::InvalidFormat(
            "L1 node section length mismatch".to_owned(),
        ));
    }
    let mut nodes = Vec::with_capacity(count);
    for _ in 0..count {
        nodes.push(L1Node {
            index: reader.index()?,
            flags: reader.u16()?,
            floor_normal_class: reader.u8()?,
            hazard_severity: reader.u8()?,
            clearance_height: reader.u16()?,
            hazard_types: reader.u16()?,
            hazard_clearance: reader.i32()?,
            cost_to_safety: reader.u32()?,
            region_id: reader.u32()?,
            confidence: reader.u16()?,
            evidence: reader.u16()?,
            contents_flags: reader.u32()?,
        });
    }
    reader.finish("L1 nodes")?;
    Ok(nodes)
}

fn encode_graph(graph: &L1Graph) -> Vec<u8> {
    let mut output = Vec::with_capacity(8 + graph.offsets().len() * 4 + graph.edges().len() * 28);
    push_u64(&mut output, graph.offsets().len() as u64);
    for offset in graph.offsets() {
        push_u32(&mut output, *offset);
    }
    for edge in graph.edges() {
        push_u32(&mut output, edge.target);
        output.push(edge.edge_type as u8);
        output.push(edge.stance as u8);
        push_u16(&mut output, edge.flags);
        push_u32(&mut output, edge.blocker);
        push_u32(&mut output, edge.cost);
        push_u16(&mut output, edge.risk);
        push_u16(&mut output, edge.confidence);
        push_u16(&mut output, edge.evidence);
        push_u16(&mut output, edge.validation_version);
        push_u32(&mut output, edge.auxiliary);
    }
    output
}

fn decode_graph(
    mut reader: Reader<'_>,
    node_count: usize,
    edge_count: usize,
) -> AtlasResult<(Vec<u32>, Vec<EdgeRecord>)> {
    let offset_count = usize::try_from(reader.u64()?)
        .map_err(|_| AtlasError::LimitExceeded("CSR offset count exceeds usize".to_owned()))?;
    if offset_count != node_count + 1 {
        return Err(AtlasError::InvalidFormat(
            "CSR offset count does not equal nodes+1".to_owned(),
        ));
    }
    let expected = offset_count
        .checked_mul(4)
        .and_then(|bytes| {
            edge_count
                .checked_mul(EdgeRecord::ENCODED_BYTES)
                .and_then(|edges| bytes.checked_add(edges))
        })
        .ok_or_else(|| AtlasError::LimitExceeded("CSR byte count overflow".to_owned()))?;
    if reader.remaining() != expected {
        return Err(AtlasError::InvalidFormat(
            "CSR section length mismatch".to_owned(),
        ));
    }
    let mut offsets = Vec::with_capacity(offset_count);
    for _ in 0..offset_count {
        offsets.push(reader.u32()?);
    }
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        edges.push(EdgeRecord {
            target: reader.u32()?,
            edge_type: EdgeType::from_u8(reader.u8()?)?,
            stance: Stance::from_u8(reader.u8()?)?,
            flags: reader.u16()?,
            blocker: reader.u32()?,
            cost: reader.u32()?,
            risk: reader.u16()?,
            confidence: reader.u16()?,
            evidence: reader.u16()?,
            validation_version: reader.u16()?,
            auxiliary: reader.u32()?,
        });
    }
    reader.finish("L1 CSR")?;
    Ok((offsets, edges))
}

fn encode_aggregate_cells(cells: &[AtlasAggregateCell]) -> Vec<u8> {
    let mut output = Vec::with_capacity(cells.len() * AGGREGATE_BYTES);
    for cell in cells {
        push_index(&mut output, cell.index);
        push_u32(&mut output, cell.contents_flags);
        push_u16(&mut output, cell.hazard_types);
        output.push(cell.hazard_severity);
        output.push(u8::from(cell.standing_passable) | (u8::from(cell.crouched_passable) << 1));
        push_u16(&mut output, cell.clearance);
        push_u32(&mut output, cell.cost_to_safety);
        push_u16(&mut output, cell.confidence);
    }
    output
}

fn decode_aggregate_cells(
    mut reader: Reader<'_>,
    count: usize,
    level: &str,
) -> AtlasResult<Vec<AtlasAggregateCell>> {
    if reader.remaining()
        != count
            .checked_mul(AGGREGATE_BYTES)
            .ok_or_else(|| AtlasError::LimitExceeded(format!("{level} byte count overflow")))?
    {
        return Err(AtlasError::InvalidFormat(format!(
            "{level} section length mismatch"
        )));
    }
    let mut cells = Vec::with_capacity(count);
    for _ in 0..count {
        let index = reader.index()?;
        let contents_flags = reader.u32()?;
        let hazard_types = reader.u16()?;
        let hazard_severity = reader.u8()?;
        let passability = reader.u8()?;
        if passability & !0x03 != 0 {
            return Err(AtlasError::InvalidFormat(format!(
                "{level} cell has unknown passability flags"
            )));
        }
        cells.push(AtlasAggregateCell {
            index,
            contents_flags,
            hazard_types,
            hazard_severity,
            standing_passable: passability & 0x01 != 0,
            crouched_passable: passability & 0x02 != 0,
            clearance: reader.u16()?,
            cost_to_safety: reader.u32()?,
            confidence: reader.u16()?,
        });
    }
    reader.finish(level)?;
    Ok(cells)
}

fn push_index(output: &mut Vec<u8>, index: GridIndex) {
    push_i32(output, index.x);
    push_i32(output, index.y);
    push_i32(output, index.z);
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

#[derive(Clone, Copy)]
struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn remaining(self) -> usize {
        self.bytes.len() - self.cursor
    }

    fn is_empty(self) -> bool {
        self.cursor == self.bytes.len()
    }

    fn take(&mut self, length: usize) -> AtlasResult<&'a [u8]> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or_else(|| AtlasError::InvalidFormat("Atlas cursor overflow".to_owned()))?;
        let result = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| AtlasError::InvalidFormat("truncated Atlas payload".to_owned()))?;
        self.cursor = end;
        Ok(result)
    }

    fn subreader(&mut self, length: usize) -> AtlasResult<Self> {
        Ok(Self::new(self.take(length)?))
    }

    fn finish(self, section: &str) -> AtlasResult<()> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(AtlasError::InvalidFormat(format!(
                "trailing bytes in {section} section"
            )))
        }
    }

    fn count(&mut self, maximum: usize, name: &str) -> AtlasResult<usize> {
        let count = usize::try_from(self.u64()?)
            .map_err(|_| AtlasError::LimitExceeded(format!("{name} count exceeds usize")))?;
        if count > maximum {
            return Err(AtlasError::LimitExceeded(format!(
                "{name} {count} > {maximum}"
            )));
        }
        Ok(count)
    }

    fn u8(&mut self) -> AtlasResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> AtlasResult<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> AtlasResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> AtlasResult<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> AtlasResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> AtlasResult<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn index(&mut self) -> AtlasResult<GridIndex> {
        Ok(GridIndex::new(self.i32()?, self.i32()?, self.i32()?))
    }
}
