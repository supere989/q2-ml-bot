use std::collections::BTreeMap;
use std::time::Instant;

use q2_lattice_rs::atlas::{
    ATLAS_CELL_SIZES, ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, ArtifactManifest, AtlasAggregateCell,
    AtlasArtifact, AtlasCounts, AtlasLevel, AtlasLimits, AtlasManifest, AtlasOrigin, BspIdentity,
    COST_INFINITY, ChannelManifest, ConservativeChild, CorridorWitness, EdgeInput, EdgeType,
    GridIndex, GridManifest, HullManifest, L0Address, L0BitPlane, L0Chunk, L0ScalarPlane, L1Graph,
    L1Node, ManifestBudgets, NodeFlags, SparseL0, Stance, ToolIdentity, aggregate_conservative,
    decode_zstd_envelope, encode_zstd_envelope, sha256_hex,
};

fn node(index: GridIndex) -> L1Node {
    L1Node {
        index,
        flags: NodeFlags::STANDING_CLEAR
            | NodeFlags::CROUCHED_CLEAR
            | NodeFlags::SAFE_TO_STAND
            | NodeFlags::SUPPORTED_FLOOR
            | NodeFlags::STANDING_PASSABLE
            | NodeFlags::CROUCHED_PASSABLE,
        floor_normal_class: 1,
        clearance_height: 96,
        hazard_types: 0,
        hazard_severity: 0,
        hazard_clearance: 8 * 256,
        cost_to_safety: 256,
        region_id: 7,
        confidence: u16::MAX,
        evidence: 1,
        contents_flags: 0,
    }
}

fn edge(source: GridIndex, target: GridIndex, edge_type: EdgeType) -> EdgeInput {
    EdgeInput {
        source,
        target,
        edge_type,
        stance: Stance::Standing,
        flags: 0,
        blocker: 0,
        cost: 256,
        risk: 0,
        confidence: u16::MAX,
        evidence: 1,
        validation_version: 1,
        auxiliary: u32::MAX,
    }
}

fn artifact(reverse_insert: bool) -> AtlasArtifact {
    let limits = AtlasLimits::default();
    let mut chunks = SparseL0::new();
    let mut source = vec![GridIndex::new(2, -1, 0), GridIndex::new(-3, 4, -2)];
    if reverse_insert {
        source.reverse();
    }
    for key in source {
        let cell = 17 + key.x.unsigned_abs() as usize;
        let mut chunk = L0Chunk::new(key);
        chunk.set_bit(L0BitPlane::Solid, cell, true).unwrap();
        chunk
            .set_bit(L0BitPlane::StandingForbiddenOrigin, 1024, true)
            .unwrap();
        chunk
            .set_scalar(L0ScalarPlane::Confidence, 17, 240)
            .unwrap();
        chunks.insert(chunk, &limits).unwrap();
    }
    let a = GridIndex::new(-1, 0, -2);
    let b = GridIndex::new(0, 0, -2);
    let c = GridIndex::new(0, 0, 1);
    let nodes = if reverse_insert {
        vec![node(c), node(a), node(b)]
    } else {
        vec![node(b), node(c), node(a)]
    };
    let edges = if reverse_insert {
        vec![edge(a, c, EdgeType::Jump), edge(a, b, EdgeType::Walk)]
    } else {
        vec![edge(a, b, EdgeType::Walk), edge(a, c, EdgeType::Jump)]
    };
    AtlasArtifact {
        origin: AtlasOrigin::snapped([-1, 0, 511]).unwrap(),
        l0: chunks,
        l1: L1Graph::build(nodes, edges, &limits).unwrap(),
        l2: vec![AtlasAggregateCell {
            index: GridIndex::new(-1, 0, 0),
            contents_flags: 3,
            hazard_types: 2,
            hazard_severity: 9,
            standing_passable: true,
            crouched_passable: true,
            clearance: 72,
            cost_to_safety: 512,
            confidence: 60_000,
        }],
        l3: vec![],
    }
}

#[test]
fn snapped_origin_and_signed_floor_are_mathematical_at_every_level() {
    assert_eq!(
        AtlasOrigin::snapped([-1, -256, -257]).unwrap().0,
        [-256, -256, -512]
    );
    let origin = AtlasOrigin([0, 0, 0]);
    for level in AtlasLevel::ALL {
        let size = level.cell_size();
        for world in -1025_i64..=1025 {
            let actual = origin.index_integer([world, world, world], level).unwrap();
            let expected = world.div_euclid(size) as i32;
            assert_eq!(actual, GridIndex::new(expected, expected, expected));
            let center = origin.center(actual, level);
            assert!(center[0] >= (actual.x as i64 * size) as f64);
            assert!(center[0] < ((actual.x as i64 + 1) * size) as f64);
        }
    }
    assert_eq!(
        origin
            .index([-0.001, -4.001, -16.001], AtlasLevel::L0)
            .unwrap(),
        GridIndex::new(-1, -2, -5)
    );
    for world in -1025_i64..=1025 {
        let indices = AtlasLevel::ALL.map(|level| {
            origin
                .index_integer([world, -world, world / 3], level)
                .unwrap()
        });
        assert_eq!(indices[0].parent(), indices[1]);
        assert_eq!(indices[1].parent(), indices[2]);
        assert_eq!(indices[2].parent(), indices[3]);
    }
    assert!(origin.index([f64::NAN, 0.0, 0.0], AtlasLevel::L0).is_err());
    assert!(
        origin
            .index_integer([i64::MAX, 0, 0], AtlasLevel::L0)
            .is_err()
    );
}

#[test]
fn parent_child_properties_hold_for_negative_nonmultiples() {
    for z in -33..=33 {
        for y in -17..=17 {
            for x in -19..=19 {
                let child = GridIndex::new(x, y, z);
                let parent = child.parent();
                assert_eq!(parent.x, x.div_euclid(4));
                assert_eq!(parent.y, y.div_euclid(4));
                assert_eq!(parent.z, z.div_euclid(4));
                let minimum = parent.child_min().unwrap();
                assert!((minimum.x..minimum.x + 4).contains(&x));
                assert!((minimum.y..minimum.y + 4).contains(&y));
                assert!((minimum.z..minimum.z + 4).contains(&z));
            }
        }
    }
    for parent in [GridIndex::new(-2, -1, -3), GridIndex::new(0, 0, 0)] {
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(parent.child(x, y, z).unwrap().parent(), parent);
                }
            }
        }
    }
}

#[test]
fn l0_chunk_address_uses_euclidean_remainder() {
    let address = L0Address::from_l0_index(GridIndex::new(-1, -16, -17));
    assert_eq!(address.chunk, GridIndex::new(-1, -1, -2));
    assert_eq!(address.local, [15, 0, 15]);
    assert_eq!(address.linear, 15 + 15 * 256);
    let origin = AtlasOrigin([-256, 0, 256]);
    for world in -513_i64..=513 {
        let l0 = origin
            .index_integer([world, -world, world / 2], AtlasLevel::L0)
            .unwrap();
        let l2 = origin
            .index_integer([world, -world, world / 2], AtlasLevel::L2)
            .unwrap();
        assert_eq!(L0Address::from_l0_index(l0).chunk, l2);
    }
}

#[test]
fn l0_is_sparse_and_enforces_chunk_and_payload_guards() {
    let mut l0 = SparseL0::new();
    assert!(l0.is_empty());
    assert!(
        l0.insert(L0Chunk::new(GridIndex::default()), &AtlasLimits::default())
            .is_err()
    );

    let mut chunk = L0Chunk::new(GridIndex::new(-1, 0, 0));
    chunk.set_bit(L0BitPlane::Lava, 4095, true).unwrap();
    chunk
        .set_scalar(L0ScalarPlane::HazardSeverity, 4095, 255)
        .unwrap();
    assert!(chunk.bit(L0BitPlane::Lava, 4095).unwrap());
    assert_eq!(
        chunk.scalar(L0ScalarPlane::HazardSeverity, 4095).unwrap(),
        255
    );

    let too_small = AtlasLimits {
        max_l0_decompressed_bytes: 100,
        ..AtlasLimits::default()
    };
    assert!(l0.insert(chunk.clone(), &too_small).is_err());
    l0.insert(chunk, &AtlasLimits::default()).unwrap();
    assert_eq!(l0.len(), 1);

    let one_chunk = AtlasLimits {
        max_l0_chunks: 1,
        ..AtlasLimits::default()
    };
    let mut another = L0Chunk::new(GridIndex::new(0, 0, 0));
    another.set_bit(L0BitPlane::Solid, 0, true).unwrap();
    assert!(l0.insert(another, &one_chunk).is_err());
}

#[test]
fn l1_graph_is_deterministic_zyx_csr() {
    let limits = AtlasLimits::default();
    let a = GridIndex::new(10, 0, -1);
    let b = GridIndex::new(-5, 2, -1);
    let c = GridIndex::new(0, 0, 0);
    let graph = L1Graph::build(
        vec![node(c), node(a), node(b)],
        vec![edge(a, c, EdgeType::Jump), edge(a, b, EdgeType::Walk)],
        &limits,
    )
    .unwrap();
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>(),
        vec![a, b, c]
    );
    assert_eq!(graph.offsets(), &[0, 2, 2, 2]);
    let outgoing = graph.outgoing(0).unwrap();
    assert_eq!(outgoing[0].edge_type, EdgeType::Walk);
    assert_eq!(outgoing[1].edge_type, EdgeType::Jump);

    let duplicate = edge(a, b, EdgeType::Walk);
    assert!(L1Graph::build(vec![node(a), node(b)], vec![duplicate, duplicate], &limits).is_err());
}

#[test]
fn conservative_aggregation_never_hides_hazard_or_blockage() {
    let base = ConservativeChild {
        contents_flags: 1,
        hazard_types: 0,
        hazard_severity: 0,
        clearance: 128,
        cost_to_safety: 1024,
        confidence: u16::MAX,
        standing_passable: true,
        crouched_passable: true,
        standing_reachable: true,
        crouched_reachable: true,
    };
    for hazard_index in 0..64 {
        let mut children = vec![base; 64];
        children[hazard_index].contents_flags = 0x80;
        children[hazard_index].hazard_types = 0x20;
        children[hazard_index].hazard_severity = 250;
        children[hazard_index].clearance = 3;
        children[hazard_index].confidence = 17;
        let all = CorridorWitness {
            child_indices: (0..64).collect(),
            reaches_boundary: true,
        };
        let aggregate = aggregate_conservative(&children, Some(&all), Some(&all)).unwrap();
        assert_eq!(aggregate.contents_flags & 0x80, 0x80);
        assert_eq!(aggregate.hazard_types & 0x20, 0x20);
        assert_eq!(aggregate.hazard_severity, 250);
        assert_eq!(aggregate.clearance, 3);
        assert_eq!(aggregate.confidence, 17);
        assert!(aggregate.standing_passable);

        children[hazard_index].standing_passable = false;
        let aggregate = aggregate_conservative(&children, Some(&all), Some(&all)).unwrap();
        assert!(!aggregate.standing_passable);
        assert!(aggregate.crouched_passable);
    }
    let unreachable = vec![
        ConservativeChild {
            cost_to_safety: 2,
            standing_reachable: false,
            crouched_reachable: false,
            ..base
        };
        64
    ];
    assert_eq!(
        aggregate_conservative(&unreachable, None, None)
            .unwrap()
            .cost_to_safety,
        COST_INFINITY
    );
    assert!(aggregate_conservative(&unreachable[..63], None, None).is_err());
}

#[test]
fn canonical_serialization_is_order_independent_and_round_trips() {
    let limits = AtlasLimits::default();
    let first = artifact(false).encode_uncompressed(&limits).unwrap();
    let second = artifact(true).encode_uncompressed(&limits).unwrap();
    assert_eq!(first, second);
    assert_eq!(&first[..8], ATLAS_MAGIC);
    assert_eq!(
        sha256_hex(&first),
        "2577be4ec3f813ce001bd2bd82de39722d1631f787301a993de89b99bc3a9774"
    );
    let restored = AtlasArtifact::decode_uncompressed(&first, &limits).unwrap();
    assert_eq!(restored.encode_uncompressed(&limits).unwrap(), first);

    let envelope = encode_zstd_envelope(&first, &limits).unwrap();
    assert_eq!(decode_zstd_envelope(&envelope, &limits).unwrap(), first);
    assert_eq!(
        AtlasArtifact::decode_zstd(&envelope, &limits).unwrap(),
        restored
    );
    assert_eq!(restored.encode_zstd(&limits).unwrap(), envelope);
}

#[test]
fn empty_atlas_has_a_canonical_round_trip() {
    let limits = AtlasLimits::default();
    let artifact = AtlasArtifact::empty(AtlasOrigin([0, 0, 0]));
    let bytes = artifact.encode_uncompressed(&limits).unwrap();
    assert_eq!(bytes.len(), 148); // 136-byte header plus one empty CSR offset.
    let restored = AtlasArtifact::decode_uncompressed(&bytes, &limits).unwrap();
    assert_eq!(restored, artifact);
    assert_eq!(restored.encode_uncompressed(&limits).unwrap(), bytes);
}

#[test]
fn corrupt_oversized_and_mixed_schema_payloads_fail_closed() {
    let limits = AtlasLimits::default();
    let valid = artifact(false).encode_uncompressed(&limits).unwrap();

    let mut mixed = valid.clone();
    mixed[8..10].copy_from_slice(&(ATLAS_SCHEMA_VERSION + 1).to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&mixed, &limits).is_err());
    assert!(AtlasArtifact::decode_uncompressed(&valid[..valid.len() - 1], &limits).is_err());

    let mut oversized = valid.clone();
    // Header offset 56 is the first u64 count (L0 chunks).
    oversized[56..64].copy_from_slice(&((limits.max_l0_chunks + 1) as u64).to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&oversized, &limits).is_err());

    let envelope = encode_zstd_envelope(&valid, &limits).unwrap();
    let mut corrupt_digest = envelope.clone();
    corrupt_digest[32] ^= 0xff;
    assert!(decode_zstd_envelope(&corrupt_digest, &limits).is_err());
    assert!(decode_zstd_envelope(&envelope[..envelope.len() - 1], &limits).is_err());

    let mut tiny = limits.clone();
    tiny.max_atlas_decompressed_bytes = 64;
    assert!(encode_zstd_envelope(&valid, &tiny).is_err());
    assert!(decode_zstd_envelope(&envelope, &tiny).is_err());
    let mut no_resident_capacity = limits.clone();
    no_resident_capacity.max_atlas_resident_bytes = 0;
    assert!(artifact(false).validate(&no_resident_capacity).is_err());
}

fn identity(name: &str, byte: u8) -> ToolIdentity {
    ToolIdentity {
        name: name.to_owned(),
        version: "1".to_owned(),
        sha256: format!("{byte:02x}").repeat(32),
    }
}

fn manifest(payload: &[u8], compressed_size: u64) -> AtlasManifest {
    let limits = AtlasLimits::default();
    let mut counts = BTreeMap::new();
    counts.insert("l0_chunks".to_owned(), 2);
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "fixture.atlas.bin.zst".to_owned(),
        ArtifactManifest::from_uncompressed(
            "application/vnd.q2.atlas-v1",
            payload,
            compressed_size,
            counts,
        ),
    );
    AtlasManifest {
        schema_version: ATLAS_SCHEMA_VERSION,
        byte_order: "little".to_owned(),
        atlas_magic: String::from_utf8_lossy(ATLAS_MAGIC).into_owned(),
        specification_sha256: "ab".repeat(32),
        bsp: BspIdentity {
            canonical_map_id: "fixture".to_owned(),
            sha256: "cd".repeat(32),
            size_bytes: 1024,
            ibsp_version: 38,
        },
        analyzer: identity("q2-atlas", 1),
        collision_oracle: identity("q2-cm-oracle", 2),
        pmove_oracle: identity("q2-pmove-oracle", 3),
        hook_oracle: identity("q2-hook-oracle", 4),
        generator: None,
        grid: GridManifest {
            origin: [-256, 0, 256],
            model0_mins: [-1, 0, 300],
            model0_maxs: [300, 600, 900],
            cell_sizes: ATLAS_CELL_SIZES.map(|value| value as u32),
            l0_chunk_dimensions: [16, 16, 16],
        },
        player_hulls: vec![
            HullManifest {
                name: "standing".to_owned(),
                mins: [-16, -16, -24],
                maxs: [16, 16, 32],
            },
            HullManifest {
                name: "crouched".to_owned(),
                mins: [-16, -16, -24],
                maxs: [16, 16, 4],
            },
        ],
        channels: vec![
            ChannelManifest {
                store: "Atlas".to_owned(),
                level: 1,
                name: "clearance".to_owned(),
                encoding: "u16".to_owned(),
                persistence: "map-static".to_owned(),
            },
            ChannelManifest {
                store: "Atlas".to_owned(),
                level: 0,
                name: "solid".to_owned(),
                encoding: "bitplane".to_owned(),
                persistence: "map-static".to_owned(),
            },
        ],
        artifacts,
        counts: AtlasCounts {
            l0_chunks: 2,
            l1_nodes: 3,
            l1_edges: 2,
            l2_cells: 1,
            l3_cells: 0,
        },
        budgets: ManifestBudgets::from(&limits),
        build_peak_rss_bytes: 64 * 1024 * 1024,
        limitations: vec!["zero hook edges".to_owned(), "fixture only".to_owned()],
        confidence_summary: "oracle-backed fixture".to_owned(),
    }
}

#[test]
fn manifest_is_canonical_bound_and_strict() {
    let limits = AtlasLimits::default();
    let payload = artifact(false).encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&payload, &limits).unwrap();
    let source = manifest(&payload, envelope.len() as u64);
    let bytes = source.canonical_json(&limits).unwrap();
    assert_eq!(
        sha256_hex(&bytes),
        "62a077734880f72b11a77bb424cae8e8a4f24a53afeaacd1687a9b8c6d2c143f"
    );
    let restored = AtlasManifest::from_canonical_json(&bytes, &limits).unwrap();
    assert_eq!(restored.channels[0].level, 0);
    assert_eq!(
        restored.limitations,
        vec!["fixture only", "zero hook edges"]
    );
    assert_eq!(restored.canonical_json(&limits).unwrap(), bytes);
    restored
        .verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &artifact(false), &limits)
        .unwrap();
    let mut tampered_payload = payload.clone();
    tampered_payload[136] ^= 1;
    assert!(
        restored
            .verify_atlas_artifact(
                "fixture.atlas.bin.zst",
                &tampered_payload,
                &artifact(false),
                &limits,
            )
            .is_err()
    );

    let pretty = serde_json::to_vec_pretty(&restored).unwrap();
    assert!(AtlasManifest::from_canonical_json(&pretty, &limits).is_err());
    let mut mixed = restored.clone();
    mixed.schema_version += 1;
    assert!(mixed.canonical_json(&limits).is_err());
    let mut bad_digest = restored.clone();
    bad_digest.pmove_oracle.sha256 = "ABC".to_owned();
    assert!(bad_digest.canonical_json(&limits).is_err());
    let mut duplicate_channel = restored.clone();
    duplicate_channel
        .channels
        .push(duplicate_channel.channels[0].clone());
    assert!(duplicate_channel.canonical_json(&limits).is_err());
    let mut out_of_range = restored.clone();
    out_of_range.grid.origin = [0, 0, 0];
    out_of_range.grid.model0_mins = [0, 0, 0];
    out_of_range.grid.model0_maxs = [i64::MAX, 1, 1];
    assert!(out_of_range.canonical_json(&limits).is_err());
    let mut over_rss = restored;
    over_rss.build_peak_rss_bytes = limits.max_build_rss_bytes + 1;
    assert!(over_rss.canonical_json(&limits).is_err());
}

#[test]
#[ignore = "manual release-mode storage benchmark"]
fn benchmark_sparse_storage_envelope_and_load() {
    let limits = AtlasLimits::default();
    let started = Instant::now();
    let mut l0 = SparseL0::new();
    for ordinal in 0..limits.max_l0_chunks {
        let mut chunk = L0Chunk::new(GridIndex::new(
            (ordinal % 40) as i32 - 20,
            ((ordinal / 40) % 30) as i32 - 15,
            (ordinal / 1200) as i32,
        ));
        for plane in [
            L0BitPlane::Solid,
            L0BitPlane::StandingForbiddenOrigin,
            L0BitPlane::CrouchedForbiddenOrigin,
        ] {
            chunk.set_bit(plane, ordinal % 4096, true).unwrap();
        }
        chunk
            .set_scalar(L0ScalarPlane::HazardSeverity, ordinal % 4096, 200)
            .unwrap();
        l0.insert(chunk, &limits).unwrap();
    }
    let build_elapsed = started.elapsed();
    let artifact = AtlasArtifact {
        origin: AtlasOrigin([0, 0, 0]),
        l0,
        l1: L1Graph::empty(),
        l2: Vec::new(),
        l3: Vec::new(),
    };
    let started = Instant::now();
    let bytes = artifact.encode_uncompressed(&limits).unwrap();
    let encode_elapsed = started.elapsed();
    let started = Instant::now();
    let envelope = encode_zstd_envelope(&bytes, &limits).unwrap();
    let compress_elapsed = started.elapsed();
    let started = Instant::now();
    let decoded = decode_zstd_envelope(&envelope, &limits).unwrap();
    let decompress_elapsed = started.elapsed();
    let started = Instant::now();
    let restored = AtlasArtifact::decode_uncompressed(&decoded, &limits).unwrap();
    let decode_elapsed = started.elapsed();
    assert_eq!(restored.l0.len(), limits.max_l0_chunks);
    eprintln!(
        "sparse_chunks={} raw_bytes={} zstd_bytes={} build_us={} encode_us={} compress_us={} decompress_us={} decode_us={}",
        restored.l0.len(),
        bytes.len(),
        envelope.len(),
        build_elapsed.as_micros(),
        encode_elapsed.as_micros(),
        compress_elapsed.as_micros(),
        decompress_elapsed.as_micros(),
        decode_elapsed.as_micros(),
    );
}
