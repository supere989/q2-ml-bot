use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

use q2_lattice_rs::atlas::{AtlasOrigin, GridIndex};
use q2_lattice_rs::dynstate::{
    BatchExpectation, DYN_FEATURE_NAMES, DYN_L2_CELL_SIZE, DYN_L3_CELL_SIZE, DYN_MAGIC, DynBatch,
    DynCell, DynError, DynFeatureIndex, DynFeatureInput, DynFence, DynLimits, DynState,
    PersistentChannels, RETIRED_DYN_MAGIC, THERMAL_MAX_AGE_TICKS, ThermalOverlay, decode_snapshot,
    encode_snapshot,
};
use sha2::{Digest, Sha256};

const HEADER_BYTES: usize = 208;

fn fence(epoch: u64) -> DynFence {
    DynFence {
        atlas_sha256: [0xa5; 32],
        map_sha256: [0x5a; 32],
        origin: AtlasOrigin([-256, -512, 0]),
        map_epoch: epoch,
    }
}

fn cell(seed: f32) -> DynCell {
    DynCell::new(
        PersistentChannels {
            engagement: seed,
            threat: seed + 1.0,
            opportunity: seed + 2.0,
            self_fire: seed + 3.0,
            deaths: seed + 4.0,
        },
        seed + 5.0,
        0.75,
    )
    .unwrap()
}

fn state(client_id: u32, client_count: u32, steps: u64) -> DynState {
    DynState::new(
        fence(17),
        client_id,
        client_count,
        steps,
        &DynLimits::default(),
    )
    .unwrap()
}

fn populated_state(client_id: u32, client_count: u32, steps: u64, count: i32) -> DynState {
    let limits = DynLimits::default();
    let binding = fence(17);
    let mut result = state(client_id, client_count, steps);
    for ordinal in 0..count {
        let x = ordinal.rem_euclid(48) - 24;
        let y = ordinal.div_euclid(48).rem_euclid(48) - 24;
        let z = ordinal.div_euclid(2304).rem_euclid(8) - 4;
        result
            .upsert_l2(
                binding,
                GridIndex::new(x, y, z),
                cell((ordinal.rem_euclid(257) + 1) as f32),
                &limits,
            )
            .unwrap();
    }
    result
}

fn feature_input(binding: DynFence) -> DynFeatureInput {
    DynFeatureInput {
        fence: binding,
        world_position: [32.0, 32.0, 32.0],
        yaw_degrees: 0.0,
        thermal: None,
        survivability: [-0.25, 0.5, 0.75],
        search_radius: 2048.0,
        score_scale: 8.0,
    }
}

#[test]
fn signed_floor_parent_property_drives_the_derived_l3_mip() {
    for coordinate in -1025_i32..=1025 {
        assert_eq!(
            GridIndex::new(coordinate, -coordinate, coordinate).parent(),
            GridIndex::new(
                coordinate.div_euclid(4),
                (-coordinate).div_euclid(4),
                coordinate.div_euclid(4),
            )
        );
    }

    let limits = DynLimits::default();
    let binding = fence(17);
    let mut dyn_state = state(0, 1, 9);
    for (ordinal, x) in [-5, -4, -3, -2, -1, 0, 1, 4].into_iter().enumerate() {
        dyn_state
            .upsert_l2(
                binding,
                GridIndex::new(x, -1, -5),
                cell(ordinal as f32 + 1.0),
                &limits,
            )
            .unwrap();
    }
    for (index, _) in dyn_state.l2_cells() {
        assert!(dyn_state.l3_cell(index.parent()).is_some());
    }
    assert!(dyn_state.l3_cell(GridIndex::new(-2, -1, -2)).is_some());
    assert!(dyn_state.l3_cell(GridIndex::new(-1, -1, -2)).is_some());
    assert!(dyn_state.l3_cell(GridIndex::new(0, -1, -2)).is_some());
    assert!(dyn_state.l3_cell(GridIndex::new(1, -1, -2)).is_some());
}

#[test]
fn l3_is_deterministic_and_has_no_independent_deposit_path() {
    let limits = DynLimits::default();
    let binding = fence(17);
    let indices = [
        GridIndex::new(-1, -1, -1),
        GridIndex::new(-4, -4, -4),
        GridIndex::new(3, 2, 1),
        GridIndex::new(4, 0, 0),
    ];
    let mut forward = state(0, 1, 10);
    let mut reverse = state(0, 1, 10);
    for (ordinal, index) in indices.into_iter().enumerate() {
        forward
            .upsert_l2(binding, index, cell(ordinal as f32 + 1.0), &limits)
            .unwrap();
    }
    for (ordinal, index) in indices.into_iter().enumerate().rev() {
        reverse
            .upsert_l2(binding, index, cell(ordinal as f32 + 1.0), &limits)
            .unwrap();
    }
    assert_eq!(
        encode_snapshot(&forward, &limits).unwrap(),
        encode_snapshot(&reverse, &limits).unwrap()
    );
}

#[test]
fn failed_l3_aggregation_is_transactional() {
    let limits = DynLimits::default();
    let binding = fence(17);
    let mut dyn_state = state(0, 1, 10);
    let first = DynCell::new(
        PersistentChannels {
            engagement: f32::MAX,
            ..PersistentChannels::ZERO
        },
        0.0,
        1.0,
    )
    .unwrap();
    dyn_state
        .upsert_l2(binding, GridIndex::new(0, 0, 0), first, &limits)
        .unwrap();
    let before = encode_snapshot(&dyn_state, &limits).unwrap();
    assert!(
        dyn_state
            .upsert_l2(binding, GridIndex::new(1, 0, 0), first, &limits)
            .is_err()
    );
    assert_eq!(dyn_state.l2_len(), 1);
    assert_eq!(encode_snapshot(&dyn_state, &limits).unwrap(), before);
}

#[test]
fn q2lat002_canonical_snapshot_round_trips_exactly() {
    let limits = DynLimits::default();
    let original = populated_state(2, 4, 65_536, 512);
    let encoded = encode_snapshot(&original, &limits).unwrap();
    assert_eq!(&encoded[..8], DYN_MAGIC);
    assert_eq!(
        u32::from_le_bytes(encoded[12..16].try_into().unwrap()),
        HEADER_BYTES as u32
    );
    assert_eq!(
        u32::from_le_bytes(encoded[160..164].try_into().unwrap()),
        DYN_L2_CELL_SIZE
    );
    assert_eq!(
        u32::from_le_bytes(encoded[164..168].try_into().unwrap()),
        DYN_L3_CELL_SIZE
    );
    let restored = decode_snapshot(&encoded, fence(17), &limits).unwrap();
    assert_eq!(restored.client_id(), 2);
    assert_eq!(restored.client_count(), 4);
    assert_eq!(restored.environment_steps(), 65_536);
    assert_eq!(restored.l2_len(), original.l2_len());
    assert_eq!(restored.l3_len(), original.l3_len());
    assert_eq!(encode_snapshot(&restored, &limits).unwrap(), encoded);
}

#[test]
fn snapshot_rejects_corruption_noncanonical_l3_and_truncation() {
    let limits = DynLimits::default();
    let original = populated_state(0, 1, 99, 32);
    let encoded = encode_snapshot(&original, &limits).unwrap();

    let mut bad_digest = encoded.clone();
    bad_digest[40] ^= 0x01;
    assert!(matches!(
        decode_snapshot(&bad_digest, fence(17), &limits),
        Err(DynError::DigestMismatch)
    ));

    let mut payload = zstd::stream::decode_all(Cursor::new(&encoded[HEADER_BYTES..])).unwrap();
    let first_l3_engagement = original.l2_len() * 40 + 12;
    let old = f32::from_le_bytes(
        payload[first_l3_engagement..first_l3_engagement + 4]
            .try_into()
            .unwrap(),
    );
    payload[first_l3_engagement..first_l3_engagement + 4]
        .copy_from_slice(&(old + 1.0).to_le_bytes());
    let recompressed = zstd::stream::encode_all(Cursor::new(&payload), 3).unwrap();
    let mut noncanonical = encoded[..HEADER_BYTES].to_vec();
    noncanonical[32..40].copy_from_slice(&(recompressed.len() as u64).to_le_bytes());
    noncanonical[40..72].copy_from_slice(&Sha256::digest(&payload));
    noncanonical.extend_from_slice(&recompressed);
    assert!(matches!(
        decode_snapshot(&noncanonical, fence(17), &limits),
        Err(DynError::InvalidFormat(message)) if message.contains("canonical derived mip")
    ));

    assert!(decode_snapshot(&encoded[..encoded.len() - 1], fence(17), &limits).is_err());
}

#[test]
fn q2lat001_and_mixed_schema_batches_fail_closed() {
    let limits = DynLimits::default();
    let valid = encode_snapshot(&state(0, 2, 128), &limits).unwrap();
    let mut retired = vec![0_u8; 16];
    retired[..8].copy_from_slice(RETIRED_DYN_MAGIC);
    assert!(matches!(
        decode_snapshot(&retired, fence(17), &limits),
        Err(DynError::RetiredSchema)
    ));
    assert!(matches!(
        DynBatch::decode(
            &[&valid, &retired],
            BatchExpectation {
                fence: fence(17),
                client_count: 2,
                environment_steps: 128,
            },
            &limits,
        ),
        Err(DynError::RetiredSchema)
    ));

    let mut future = valid.clone();
    future[8..10].copy_from_slice(&3_u16.to_le_bytes());
    assert!(matches!(
        decode_snapshot(&future, fence(17), &limits),
        Err(DynError::MixedSchema {
            expected: 2,
            found: 3
        })
    ));
}

#[test]
fn q2lat002_rejects_cell_size_mismatches_and_the_unfenced_legacy_header() {
    let limits = DynLimits::default();
    let valid = encode_snapshot(&state(0, 1, 128), &limits).unwrap();

    let mut wrong_l2 = valid.clone();
    wrong_l2[160..164].copy_from_slice(&(DYN_L2_CELL_SIZE / 2).to_le_bytes());
    assert!(matches!(
        decode_snapshot(&wrong_l2, fence(17), &limits),
        Err(DynError::InvalidFormat(message)) if message.contains("cell-size fence")
    ));

    let mut wrong_l3 = valid.clone();
    wrong_l3[164..168].copy_from_slice(&(DYN_L3_CELL_SIZE / 2).to_le_bytes());
    assert!(matches!(
        decode_snapshot(&wrong_l3, fence(17), &limits),
        Err(DynError::InvalidFormat(message)) if message.contains("cell-size fence")
    ));

    let mut legacy = valid.clone();
    legacy.drain(160..168);
    legacy[12..16].copy_from_slice(&200_u32.to_le_bytes());
    assert!(matches!(
        decode_snapshot(&legacy, fence(17), &limits),
        Err(DynError::InvalidFormat(message)) if message.contains("header size")
    ));
}

#[test]
fn epoch_atlas_map_and_environment_step_fences_reject_stale_state() {
    let limits = DynLimits::default();
    let binding = fence(17);
    let mut dyn_state = state(0, 1, 256);
    let stale_epoch = fence(16);
    assert!(matches!(
        dyn_state.upsert_l2(stale_epoch, GridIndex::new(0, 0, 0), cell(1.0), &limits),
        Err(DynError::FenceMismatch("map_epoch"))
    ));
    assert!(matches!(
        dyn_state.feature_block(feature_input(stale_epoch)),
        Err(DynError::FenceMismatch("map_epoch"))
    ));
    assert!(matches!(
        dyn_state.set_environment_steps(binding, 255),
        Err(DynError::StaleEnvironmentSteps {
            expected: 256,
            found: 255
        })
    ));

    let encoded = encode_snapshot(&dyn_state, &limits).unwrap();
    let mut wrong_atlas = binding;
    wrong_atlas.atlas_sha256 = [0x11; 32];
    assert!(matches!(
        decode_snapshot(&encoded, wrong_atlas, &limits),
        Err(DynError::FenceMismatch("atlas_sha256"))
    ));
    let mut wrong_map = binding;
    wrong_map.map_sha256 = [0x22; 32];
    assert!(matches!(
        decode_snapshot(&encoded, wrong_map, &limits),
        Err(DynError::FenceMismatch("map_sha256"))
    ));
}

#[test]
fn thermal_is_bounded_ephemeral_and_excluded_from_snapshot() {
    let limits = DynLimits::default();
    let dyn_state = populated_state(0, 1, 50, 8);
    let before = encode_snapshot(&dyn_state, &limits).unwrap();
    let mut thermal = ThermalOverlay::default();
    thermal
        .observe(0x0003_0000_0007, [128.0, -64.0, 48.0], 0.8, 50, 5)
        .unwrap();
    assert!(thermal.strongest(55).is_some());
    thermal.expire(56);
    assert!(thermal.is_empty());
    assert!(
        thermal
            .observe(9, [0.0; 3], 1.0, 50, THERMAL_MAX_AGE_TICKS + 1)
            .is_err()
    );
    assert_eq!(encode_snapshot(&dyn_state, &limits).unwrap(), before);
    assert!(
        !before
            .windows(8)
            .any(|window| window == 0x0003_0000_0007_u64.to_le_bytes())
    );
}

#[test]
fn frozen_named_features_use_yaw_local_quake_right_and_thermal_override() {
    let limits = DynLimits::default();
    let binding = DynFence {
        atlas_sha256: [0x33; 32],
        map_sha256: [0x44; 32],
        origin: AtlasOrigin([0, 0, 0]),
        map_epoch: 8,
    };
    let mut dyn_state = DynState::new(binding, 0, 1, 100, &limits).unwrap();
    dyn_state
        .upsert_l2(binding, GridIndex::new(0, 0, 0), cell(4.0), &limits)
        .unwrap();
    dyn_state
        .upsert_l2(
            binding,
            GridIndex::new(0, -1, 0),
            DynCell::new(
                PersistentChannels {
                    engagement: 8.0,
                    threat: 80.0,
                    opportunity: 0.0,
                    self_fire: 0.0,
                    deaths: 2.0,
                },
                4.0,
                1.0,
            )
            .unwrap(),
            &limits,
        )
        .unwrap();

    let mut input = feature_input(binding);
    let persistent = dyn_state.feature_block(input).unwrap();
    assert_eq!(DYN_FEATURE_NAMES.len(), 24);
    assert!(persistent.values[DynFeatureIndex::CurrentEngagement as usize] > 0.0);
    assert_eq!(
        persistent.values[DynFeatureIndex::CurrentConfidence as usize],
        0.75
    );
    assert!(persistent.values[DynFeatureIndex::CombatThreatQuakeRight as usize] > 0.0);
    assert!(persistent.nearest_death_score > 0.0);
    assert_eq!(
        persistent.values[DynFeatureIndex::WinMargin as usize],
        -0.25
    );
    assert_eq!(
        persistent.values[DynFeatureIndex::EffectiveHealthNorm as usize],
        0.5
    );
    assert_eq!(
        persistent.values[DynFeatureIndex::OwnDpsShare as usize],
        0.75
    );

    let mut thermal = ThermalOverlay::default();
    thermal
        .observe(7, [160.0, 32.0, 32.0], 0.625, 100, 5)
        .unwrap();
    input.thermal = thermal.strongest(100);
    let live = dyn_state.feature_block(input).unwrap();
    assert!(live.values[DynFeatureIndex::ImmediateThermalForward as usize] > 0.0);
    assert_eq!(
        live.values[DynFeatureIndex::ImmediateThermalQuakeRight as usize],
        0.0
    );
    assert_eq!(
        live.values[DynFeatureIndex::ImmediateThermalHeat as usize],
        0.625
    );
}

#[test]
fn strict_cell_resident_snapshot_and_batch_limits_are_enforced() {
    let limits = DynLimits {
        max_materialized_cells: 2,
        max_l2_cells: 2,
        max_l3_cells: 2,
        ..DynLimits::default()
    };
    let binding = fence(17);
    let mut dyn_state = DynState::new(binding, 0, 1, 0, &limits).unwrap();
    dyn_state
        .upsert_l2(binding, GridIndex::new(0, 0, 0), cell(1.0), &limits)
        .unwrap();
    assert!(
        dyn_state
            .upsert_l2(binding, GridIndex::new(4, 0, 0), cell(1.0), &limits)
            .is_err()
    );
    assert_eq!(dyn_state.l2_len(), 1);
    assert_eq!(dyn_state.l3_len(), 1);

    let default_limits = DynLimits::default();
    let dyn_states: Vec<_> = (0..4)
        .map(|client_id| populated_state(client_id, 4, 4096, 18_432))
        .collect();
    let combined_resident: usize = dyn_states
        .iter()
        .map(DynState::resident_bytes_estimate)
        .sum();
    eprintln!("four_client_resident_bytes={combined_resident}");
    assert!(combined_resident < 8 * 1024 * 1024);
    let snapshots: Vec<_> = dyn_states
        .iter()
        .map(|dyn_state| encode_snapshot(dyn_state, &default_limits).unwrap())
        .collect();
    let combined: usize = snapshots.iter().map(Vec::len).sum();
    eprintln!("four_snapshot_compressed_bytes={combined}");
    assert!(combined < 8 * 1024 * 1024);
    let refs: Vec<_> = snapshots.iter().map(Vec::as_slice).collect();
    let batch = DynBatch::decode(
        &refs,
        BatchExpectation {
            fence: binding,
            client_count: 4,
            environment_steps: 4096,
        },
        &default_limits,
    )
    .unwrap();
    assert_eq!(batch.states.len(), 4);
    assert_eq!(batch.report.compressed_bytes, combined);
    assert_eq!(batch.report.resident_bytes, combined_resident);

    let soft = DynLimits {
        batch_soft_compressed_bytes: 1,
        ..default_limits.clone()
    };
    assert!(
        DynBatch::decode(
            &refs,
            BatchExpectation {
                fence: binding,
                client_count: 4,
                environment_steps: 4096,
            },
            &soft,
        )
        .unwrap()
        .report
        .soft_limit_exceeded
    );
    let hard = DynLimits {
        batch_hard_compressed_bytes: combined - 1,
        ..default_limits
    };
    assert!(matches!(
        DynBatch::decode(
            &refs,
            BatchExpectation {
                fence: binding,
                client_count: 4,
                environment_steps: 4096,
            },
            &hard,
        ),
        Err(DynError::LimitExceeded(message)) if message.contains("hard limit")
    ));
    let resident_hard = DynLimits {
        batch_hard_resident_bytes: combined_resident - 1,
        ..DynLimits::default()
    };
    assert!(matches!(
        DynBatch::decode(
            &refs,
            BatchExpectation {
                fence: binding,
                client_count: 4,
                environment_steps: 4096,
            },
            &resident_hard,
        ),
        Err(DynError::LimitExceeded(message)) if message.contains("resident bytes")
    ));
}

#[test]
fn batch_rejects_duplicate_clients_and_stale_steps() {
    let limits = DynLimits::default();
    let first = encode_snapshot(&state(0, 2, 100), &limits).unwrap();
    let duplicate = encode_snapshot(&state(0, 2, 100), &limits).unwrap();
    assert!(matches!(
        DynBatch::decode(
            &[&first, &duplicate],
            BatchExpectation {
                fence: fence(17),
                client_count: 2,
                environment_steps: 100,
            },
            &limits,
        ),
        Err(DynError::InvalidFormat(message)) if message.contains("duplicate")
    ));
    let second = encode_snapshot(&state(1, 2, 99), &limits).unwrap();
    assert!(matches!(
        DynBatch::decode(
            &[&first, &second],
            BatchExpectation {
                fence: fence(17),
                client_count: 2,
                environment_steps: 100,
            },
            &limits,
        ),
        Err(DynError::StaleEnvironmentSteps {
            expected: 100,
            found: 99
        })
    ));
}

/// Manual gate: `cargo test --release --test dyn_snapshot
/// four_client_feature_assembly_p99_is_below_half_a_millisecond -- --ignored`.
#[test]
#[ignore = "release-mode resident-query benchmark"]
fn four_client_feature_assembly_p99_is_below_half_a_millisecond() {
    let binding = fence(17);
    let states: Vec<_> = (0..4)
        .map(|client_id| populated_state(client_id, 4, 10_000, 18_432))
        .collect();
    let inputs = [
        [-160.0, -320.0, 32.0],
        [96.0, -64.0, 96.0],
        [352.0, 192.0, 160.0],
        [608.0, 448.0, 224.0],
    ];
    for round in 0..128 {
        for (client, dyn_state) in states.iter().enumerate() {
            let mut input = feature_input(binding);
            input.world_position = inputs[(round + client) % inputs.len()];
            black_box(dyn_state.feature_block(input).unwrap());
        }
    }

    let mut samples = Vec::with_capacity(2_000);
    for round in 0..2_000 {
        let start = Instant::now();
        for (client, dyn_state) in states.iter().enumerate() {
            let mut input = feature_input(binding);
            input.world_position = inputs[(round + client) % inputs.len()];
            black_box(dyn_state.feature_block(input).unwrap());
        }
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let p99 = samples[(samples.len() * 99) / 100];
    eprintln!(
        "four_client_feature_assembly samples={} p99_us={:.3}",
        samples.len(),
        p99.as_secs_f64() * 1_000_000.0
    );
    assert!(
        p99 < Duration::from_micros(500),
        "four-client feature assembly p99 {p99:?} exceeds 0.5 ms"
    );
}
