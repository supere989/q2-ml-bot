use std::collections::BTreeMap;
use std::time::Instant;

use q2_lattice_rs::atlas::{
    ADVISORY_SPATIAL_WIDTH, ATLAS_CELL_SIZES, ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, ArtifactManifest,
    AtlasAggregateCell, AtlasArtifact, AtlasCounts, AtlasLevel, AtlasLimits, AtlasManifest,
    AtlasOrigin, AtlasRuntime, AtlasRuntimeSlot, B1AuthorityExecutables, B1AuthorityIdentities,
    B1AuthorityIdentity, B1NormativeDocuments, B1RuntimeAuthoritySeal, BspIdentity,
    COLLISION_ORACLE_NAME, COLLISION_ORACLE_SCHEMA, COST_INFINITY, ChannelManifest,
    CollisionOracleAdmission, CollisionParameters, CollisionSourceClosure, ConservativeChild,
    CorridorWitness, EdgeInput, EdgeType, FALL_ORACLE_NAME, FALL_ORACLE_SCHEMA,
    FallOracleAdmission, FallParameters, FallSourceClosure, GUIDE_FEATURE_WIDTH, GridIndex,
    GridManifest, HAZARD_CLEARANCE_UNREACHABLE_HAZARD, HAZARD_CLEARANCE_UNREACHABLE_SAFE,
    HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA, HOOK_PARITY_CASES_V1, HOOK_PARITY_NAME,
    HOOK_PARITY_SCHEMA, HazardComponentField, HookOracleAdmission, HookParameters,
    HookParityAttestation, HookSourceClosure, HullManifest, L0Address, L0BitPlane, L0Chunk,
    L0ScalarPlane, L1Graph, L1Node, MASK_PLAYERSOLID_V1, MASK_SHOT_V1, ManifestBudgets, NodeFlags,
    OBJECTIVE_MEDIA_TYPE, ORACLE_SEMANTIC_VERSION, ObjectiveBelief, ObjectiveClass,
    OracleAdmissions, OracleBspBinding, OracleToolIdentity, PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA,
    PmoveOracleAdmission, PmoveParameters, PmoveSourceClosure, PolicyHazardBits,
    RECOVERY_EVIDENCE_FIELD_NAMES, RECOVERY_FEATURE_WIDTH, RECOVERY_REPAIR_NODE_LIMIT,
    RecoveryOverlay, RecoveryQuery, SparseL0, Stance, ToolIdentity, advisory_spatial_feature_names,
    aggregate_conservative, decode_zstd_envelope, encode_zstd_envelope, install_static_costs,
    install_static_hazard_clearances, recovery_features, sha256_hex, solve_static_costs,
    solve_static_hazard_clearances, validate_static_costs, validate_static_hazard_clearances,
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

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn bsp_identity() -> BspIdentity {
    BspIdentity {
        canonical_map_id: "fixture".to_owned(),
        sha256: digest(0xcd),
        provenance_sha256: digest(0xce),
        size_bytes: 1024,
        ibsp_version: 38,
    }
}

fn oracle_tool(name: &str, schema: &str, executable: u8, physics: u8) -> OracleToolIdentity {
    OracleToolIdentity {
        name: name.to_owned(),
        schema: schema.to_owned(),
        version: ORACLE_SEMANTIC_VERSION,
        executable_sha256: digest(executable),
        tool_identity_sha256: digest(physics.wrapping_add(1)),
        physics_identity_sha256: digest(physics),
    }
}

fn oracle_binding(bsp: &BspIdentity) -> OracleBspBinding {
    OracleBspBinding {
        sha256: bsp.sha256.clone(),
        provenance_sha256: bsp.provenance_sha256.clone(),
    }
}

fn oracle_admissions(bsp: &BspIdentity, pmove: bool, hook: bool) -> OracleAdmissions {
    let mut collision_tool =
        oracle_tool(COLLISION_ORACLE_NAME, COLLISION_ORACLE_SCHEMA, 0x11, 0x12);
    collision_tool.executable_sha256 =
        "781edaee1b9317766dbf831ad5edc8b5fdebe696969ca1efe0e54e2f3e5c7d1e".to_owned();
    collision_tool.tool_identity_sha256 =
        "50fd0df0a296c54d0060dc2406977b1c78f9392ea1643e010f6759622557cfdf".to_owned();
    let collision = CollisionOracleAdmission {
        tool: collision_tool,
        bsp: oracle_binding(bsp),
        parameters: CollisionParameters {
            mask_playersolid: MASK_PLAYERSOLID_V1,
            mask_shot: MASK_SHOT_V1,
        },
        source: CollisionSourceClosure {
            collision_sha256: digest(0x13),
            shared_header_sha256: digest(0x14),
            shared_source_sha256: digest(0x15),
            build_contract: "cc-v1 -fno-strict-aliasing".to_owned(),
        },
        contract_sha256: String::new(),
    }
    .seal();
    let pmove_admission = pmove.then(|| {
        let mut tool = oracle_tool(PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA, 0x21, 0x22);
        tool.executable_sha256 = "66b481e924ec3d0a5e4eaf5458dd34cfe3c0927d5b7650455bceb368666718e4".to_owned();
        tool.tool_identity_sha256 = "50fd0df0a296c54d0060dc2406977b1c78f9392ea1643e010f6759622557cfdf".to_owned();
        PmoveOracleAdmission {
            tool,
            bsp: oracle_binding(bsp),
            parameters: PmoveParameters {
                gravity: 800,
                airaccelerate_f32_bits: 0.0_f32.to_bits(),
                constants: "stopspeed=100;maxspeed=300;duckspeed=100;accelerate=10;wateraccelerate=10;friction=6;waterfriction=1;waterspeed=400".to_owned(),
            },
            source: PmoveSourceClosure {
                collision_sha256: collision.source.collision_sha256.clone(),
                pmove_sha256: digest(0x23),
                shared_header_sha256: collision.source.shared_header_sha256.clone(),
                shared_source_sha256: collision.source.shared_source_sha256.clone(),
                build_contract: collision.source.build_contract.clone(),
            },
            contract_sha256: String::new(),
        }
        .seal()
    });
    let hook_admission = hook.then(|| {
        let pmove = pmove_admission
            .as_ref()
            .expect("hook fixture always includes companion pmove");
        let mut tool = oracle_tool(HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA, 0x31, 0x32);
        tool.executable_sha256 =
            "cd8bc4107ae2e9f4ac006fbe469b360832db80b96a5597c2e5dfe12c32dc9284".to_owned();
        tool.tool_identity_sha256 =
            "9c47e3339df3f194c7729ea95c1955708540411d8baffc8208aa92349e1d2e78".to_owned();
        tool.physics_identity_sha256 =
            "38f441106d653997466f8ace13baebe5e5515d6b77a7edf535a1d93576eef9d3".to_owned();
        let parity = HookParityAttestation {
            name: HOOK_PARITY_NAME.to_owned(),
            schema: HOOK_PARITY_SCHEMA.to_owned(),
            version: ORACLE_SEMANTIC_VERSION,
            passed: true,
            case_count: HOOK_PARITY_CASES_V1,
            fixture_bsp_sha256: digest(0x33),
            fixture_provenance_sha256: digest(0x34),
            fixture_collision_physics_identity_sha256: digest(0x35),
            fixture_pmove_physics_identity_sha256: digest(0x36),
            hook_physics_identity_sha256: tool.physics_identity_sha256.clone(),
            collision_tool_sha256: collision.tool.executable_sha256.clone(),
            pmove_tool_sha256: pmove.tool.executable_sha256.clone(),
            hook_tool_sha256: tool.executable_sha256.clone(),
            q2ded_sha256: digest(0x37),
            game_module_sha256: digest(0x38),
            transcript_sha256: digest(0x39),
            attestation_sha256: String::new(),
        }
        .seal();
        HookOracleAdmission {
            tool,
            bsp: oracle_binding(bsp),
            parameters: HookParameters {
                hook_speed_f32_bits: 900.0_f32.to_bits(),
                hook_pullspeed_f32_bits: 1700.0_f32.to_bits(),
                hook_sky: false,
                hook_maxtime_f32_bits: 5.0_f32.to_bits(),
                full_velocity_overwrite: true,
            },
            source: HookSourceClosure {
                shared_c_sha256: digest(0x3a),
                shared_h_sha256: digest(0x3b),
                integration_sha256: digest(0x3c),
                math_sha256: digest(0x3d),
                build_contract: "cc-v1 -fno-strict-aliasing".to_owned(),
            },
            parity,
            contract_sha256: String::new(),
        }
        .seal()
    });
    let pmove_tool = pmove_admission.as_ref().map(|item| &item.tool);
    let fall_constants = "player_model=255,noclip=1,grapple_fly=0,release_grace=0.2,delta_scale=0.0001,water1=0.5,water2=0.25,water3=suppress,footstep=1,short=15,damage=30,far=55,fall_value_scale=0.5,fall_value_max=40,fall_time=0.3,damage_divisor=2,df_no_falling=8";
    let fall_oracle = FallOracleAdmission {
        tool: OracleToolIdentity {
            name: FALL_ORACLE_NAME.to_owned(),
            schema: FALL_ORACLE_SCHEMA.to_owned(),
            version: ORACLE_SEMANTIC_VERSION,
            executable_sha256: "dfdcf7ed74cc3ad7b8aa73df86986a8a4a31207da98ccffb4dd61673c324bef8"
                .to_owned(),
            tool_identity_sha256:
                "8f6706edf203bb75451fd148943fb9d0425a1b112f086b6886788434973117d5".to_owned(),
            physics_identity_sha256:
                "8b1f06550cb546d329bbce209f2b13248810fc10e0e19799616a338c0f633582".to_owned(),
        },
        parameters: FallParameters {
            fall_damagemod_f32_bits: 1.0_f32.to_bits(),
            deathmatch: true,
            dmflags: 0,
            constants: fall_constants.to_owned(),
        },
        source: FallSourceClosure {
            shared_c_sha256: "6d30c143e359e18784615ad0f4b21a85b3b4b9b2d4b841792685b19a88a7b6d8"
                .to_owned(),
            shared_h_sha256: "debd12ca5315cfa9e6cff714bad2d2c2fe708e378a37776e6665793aa3967357"
                .to_owned(),
            integration_sha256: "326f59b12ee60bd93252a9d5a39428c535097ed6bc7a6258f2326a3cdb12ed62"
                .to_owned(),
            game_header_sha256: "da27f13498fb7120b037b2a6b6ce0a36f4e90a90d1caf0c09c7aaeb1c8310877"
                .to_owned(),
            constants_sha256: "6274fdec332e9d51db6f1b8ca8a836835902e5269957f62c72e3d63a3a54c703"
                .to_owned(),
            build_contract: "lithium-linux-c99-o1-f32-shared-fall-v1".to_owned(),
            tool_closure_sha256: "8f6706edf203bb75451fd148943fb9d0425a1b112f086b6886788434973117d5"
                .to_owned(),
        },
        contract_sha256: String::new(),
    }
    .seal();
    OracleAdmissions {
        b1_runtime_authority_seal: B1RuntimeAuthoritySeal {
            schema: "q2-b1-runtime-authority-seal-v1".to_owned(),
            normative_documents: B1NormativeDocuments {
                design_sha256: "c55fc7ffc32bd0e88410b8493b46c179f3333f3806632ff8e6530f1c717508e6"
                    .to_owned(),
                plan_sha256: "371577feb8c40f542c90eec4b4aa91ef84c4a8e2019bf1614e59c46aedfec410"
                    .to_owned(),
            },
            hook_parity_attestation_sha256:
                "2e473d8face6b89f5b32798ddc5264bb8cc406e8dc29fd837e85bbd11b53d5ab".to_owned(),
            fixture_bsp_sha256: "ed6c3ae52dffce93b932756486fdaea3992f6a8ce68dddf2fbfd4281e4515b3f"
                .to_owned(),
            analysis_bsp_sha256: bsp.sha256.clone(),
            executables: B1AuthorityExecutables {
                cm_sha256: collision.tool.executable_sha256.clone(),
                pmove_sha256: pmove_tool
                    .map_or_else(|| digest(0x45), |item| item.executable_sha256.clone()),
                hook_sha256: "cd8bc4107ae2e9f4ac006fbe469b360832db80b96a5597c2e5dfe12c32dc9284"
                    .to_owned(),
                fall_sha256: fall_oracle.tool.executable_sha256.clone(),
            },
            identities: B1AuthorityIdentities {
                collision: B1AuthorityIdentity {
                    tool_identity: collision.tool.tool_identity_sha256.clone(),
                    physics_identity: collision.tool.physics_identity_sha256.clone(),
                },
                pmove: B1AuthorityIdentity {
                    tool_identity:
                        "50fd0df0a296c54d0060dc2406977b1c78f9392ea1643e010f6759622557cfdf"
                            .to_owned(),
                    physics_identity: pmove_tool
                        .map_or_else(|| digest(0x4a), |item| item.physics_identity_sha256.clone()),
                },
                hook: B1AuthorityIdentity {
                    tool_identity:
                        "9c47e3339df3f194c7729ea95c1955708540411d8baffc8208aa92349e1d2e78"
                            .to_owned(),
                    physics_identity:
                        "38f441106d653997466f8ace13baebe5e5515d6b77a7edf535a1d93576eef9d3"
                            .to_owned(),
                },
                fall: B1AuthorityIdentity {
                    tool_identity: fall_oracle.tool.tool_identity_sha256.clone(),
                    physics_identity: fall_oracle.tool.physics_identity_sha256.clone(),
                },
            },
        },
        collision_oracle: collision,
        fall_oracle,
        pmove_oracle: pmove_admission,
        hook_oracle: hook_admission,
    }
}

fn artifact(reverse_insert: bool) -> AtlasArtifact {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
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
        l1: L1Graph::build(nodes, edges, &admission, &limits).unwrap(),
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
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, false).admit(&bsp).unwrap();
    let a = GridIndex::new(10, 0, -1);
    let b = GridIndex::new(-5, 2, -1);
    let c = GridIndex::new(0, 0, 0);
    let graph = L1Graph::build(
        vec![node(c), node(a), node(b)],
        vec![edge(a, c, EdgeType::Jump), edge(a, b, EdgeType::Walk)],
        &admission,
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
    assert!(
        L1Graph::build(
            vec![node(a), node(b)],
            vec![duplicate, duplicate],
            &admission,
            &limits,
        )
        .is_err()
    );
}

fn two_node_graph(
    edge_type: EdgeType,
    evidence: u16,
    validation_version: u16,
    admission: &q2_lattice_rs::atlas::EdgeAdmission,
) -> Result<L1Graph, q2_lattice_rs::atlas::AtlasError> {
    let source = GridIndex::new(0, 0, 0);
    let target = GridIndex::new(1, 0, 0);
    let mut candidate = edge(source, target, edge_type);
    candidate.evidence = evidence;
    candidate.validation_version = validation_version;
    L1Graph::build(
        vec![node(source), node(target)],
        vec![candidate],
        admission,
        &AtlasLimits::default(),
    )
}

#[test]
fn graph_trajectory_edges_require_admitted_optional_oracles() {
    let bsp = bsp_identity();
    assert!(oracle_admissions(&bsp, false, false).admit(&bsp).is_err());

    let pmove_only = oracle_admissions(&bsp, true, false).admit(&bsp).unwrap();
    assert!(two_node_graph(EdgeType::Jump, 1, 1, &pmove_only).is_ok());
    assert!(two_node_graph(EdgeType::ControlledDrop, 1, 1, &pmove_only).is_err());
    assert!(two_node_graph(EdgeType::ControlledDrop, 2, 1, &pmove_only).is_err());
    assert!(two_node_graph(EdgeType::ControlledDrop, 10, 1, &pmove_only).is_ok());
    assert!(two_node_graph(EdgeType::Ladder, 1, 1, &pmove_only).is_err());
    assert!(two_node_graph(EdgeType::Ladder, 10, 1, &pmove_only).is_err());
    assert!(two_node_graph(EdgeType::Ladder, 11, 2, &pmove_only).is_err());
    assert!(two_node_graph(EdgeType::Ladder, 11, 1, &pmove_only).is_ok());
    assert!(two_node_graph(EdgeType::Hook, 1, 1, &pmove_only).is_err());

    let full = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    assert!(two_node_graph(EdgeType::Hook, 1, 1, &full).is_ok());
}

#[test]
fn every_materialized_edge_requires_nonzero_evidence_and_validation_version() {
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    for edge_type in [
        EdgeType::Walk,
        EdgeType::StrafeWalk,
        EdgeType::Step,
        EdgeType::Jump,
        EdgeType::ControlledDrop,
        EdgeType::CrouchEnter,
        EdgeType::CrouchHold,
        EdgeType::CrouchExit,
        EdgeType::WaterTransition,
        EdgeType::Mover,
        EdgeType::Teleporter,
        EdgeType::Hook,
        EdgeType::Ladder,
    ] {
        assert!(two_node_graph(edge_type, 0, 1, &admission).is_err());
        assert!(two_node_graph(edge_type, 1, 0, &admission).is_err());
        let evidence = match edge_type {
            EdgeType::ControlledDrop => 10,
            EdgeType::Ladder => 11,
            _ => 1,
        };
        assert!(two_node_graph(edge_type, evidence, 1, &admission).is_ok());
    }
}

#[test]
fn edge_type_v1_discriminants_remain_stable_when_ladder_is_appended() {
    assert_eq!(EdgeType::Walk as u8, 0);
    assert_eq!(EdgeType::ControlledDrop as u8, 4);
    assert_eq!(EdgeType::Teleporter as u8, 10);
    assert_eq!(EdgeType::Hook as u8, 11);
    assert_eq!(EdgeType::Ladder as u8, 12);
}

#[test]
fn oracle_contract_rejects_tool_schema_version_bsp_parameter_and_source_mismatches() {
    let bsp = bsp_identity();
    let valid = oracle_admissions(&bsp, true, true);
    valid.admit(&bsp).unwrap();

    for mutation in 0..3 {
        let mut candidate = valid.clone();
        let tool = &mut candidate.collision_oracle.tool;
        match mutation {
            0 => tool.name.push_str("-wrong"),
            1 => tool.schema.push_str("-wrong"),
            _ => tool.version += 1,
        }
        assert!(candidate.admit(&bsp).is_err());
    }
    for mutation in 0..3 {
        let mut candidate = valid.clone();
        let tool = &mut candidate.pmove_oracle.as_mut().unwrap().tool;
        match mutation {
            0 => tool.name.push_str("-wrong"),
            1 => tool.schema.push_str("-wrong"),
            _ => tool.version += 1,
        }
        assert!(candidate.admit(&bsp).is_err());
    }
    for mutation in 0..3 {
        let mut candidate = valid.clone();
        let tool = &mut candidate.hook_oracle.as_mut().unwrap().tool;
        match mutation {
            0 => tool.name.push_str("-wrong"),
            1 => tool.schema.push_str("-wrong"),
            _ => tool.version += 1,
        }
        assert!(candidate.admit(&bsp).is_err());
    }

    let mut wrong_bsp = valid.clone();
    wrong_bsp.collision_oracle.bsp.sha256 = digest(0xee);
    assert!(wrong_bsp.admit(&bsp).is_err());
    let mut wrong_provenance = valid.clone();
    wrong_provenance
        .pmove_oracle
        .as_mut()
        .unwrap()
        .bsp
        .provenance_sha256 = digest(0xef);
    assert!(wrong_provenance.admit(&bsp).is_err());
    let mut wrong_hook_bsp = valid.clone();
    wrong_hook_bsp.hook_oracle.as_mut().unwrap().bsp.sha256 = digest(0xf0);
    assert!(wrong_hook_bsp.admit(&bsp).is_err());

    let mut wrong_mask = valid.clone();
    wrong_mask.collision_oracle.parameters.mask_playersolid ^= 1;
    wrong_mask.collision_oracle = wrong_mask.collision_oracle.seal();
    assert!(wrong_mask.admit(&bsp).is_err());
    let mut wrong_parameter_closure = valid.clone();
    wrong_parameter_closure
        .pmove_oracle
        .as_mut()
        .unwrap()
        .parameters
        .gravity = 799;
    assert!(wrong_parameter_closure.admit(&bsp).is_err());
    let mut illegal_airaccelerate = valid.clone();
    let pmove = illegal_airaccelerate.pmove_oracle.take().unwrap();
    let mut pmove = pmove;
    pmove.parameters.airaccelerate_f32_bits = (-1.0_f32).to_bits();
    illegal_airaccelerate.pmove_oracle = Some(pmove.seal());
    assert!(illegal_airaccelerate.admit(&bsp).is_err());
    let mut wrong_source_closure = valid.clone();
    wrong_source_closure
        .hook_oracle
        .as_mut()
        .unwrap()
        .source
        .math_sha256 = digest(0xaa);
    assert!(wrong_source_closure.admit(&bsp).is_err());
    let mut zero_pullspeed = valid.clone();
    let hook = zero_pullspeed.hook_oracle.take().unwrap();
    let mut hook = hook;
    hook.parameters.hook_pullspeed_f32_bits = 0.0_f32.to_bits();
    zero_pullspeed.hook_oracle = Some(hook.seal());
    assert!(zero_pullspeed.admit(&bsp).is_err());
    let mut wrong_tool_closure = valid.clone();
    wrong_tool_closure.collision_oracle.tool.executable_sha256 = digest(0xab);
    assert!(wrong_tool_closure.admit(&bsp).is_err());

    for field in 0..4 {
        let mut zero_contract = valid.clone();
        match field {
            0 => zero_contract.collision_oracle.contract_sha256 = "00".repeat(32),
            1 => zero_contract.pmove_oracle.as_mut().unwrap().contract_sha256 = "00".repeat(32),
            2 => zero_contract.hook_oracle.as_mut().unwrap().contract_sha256 = "00".repeat(32),
            _ => zero_contract.fall_oracle.contract_sha256 = "00".repeat(32),
        }
        assert!(zero_contract.admit(&bsp).is_err());
    }

    let mut wrong_companion_source = valid.clone();
    let pmove = wrong_companion_source.pmove_oracle.take().unwrap();
    let mut pmove = pmove;
    pmove.source.collision_sha256 = digest(0xac);
    wrong_companion_source.pmove_oracle = Some(pmove.seal());
    assert!(wrong_companion_source.admit(&bsp).is_err());

    let mut wrong_overwrite_law = valid;
    let hook = wrong_overwrite_law.hook_oracle.take().unwrap();
    let mut hook = hook;
    hook.parameters.full_velocity_overwrite = false;
    wrong_overwrite_law.hook_oracle = Some(hook.seal());
    assert!(wrong_overwrite_law.admit(&bsp).is_err());
}

#[test]
fn every_b1_seal_field_and_fall_gate_field_is_authoritative() {
    let bsp = bsp_identity();
    let valid = oracle_admissions(&bsp, true, true);
    valid.admit(&bsp).unwrap();
    for mutation in 0..18 {
        let mut candidate = valid.clone();
        let seal = &mut candidate.b1_runtime_authority_seal;
        match mutation {
            0 => seal.schema.push('x'),
            1 => seal.normative_documents.design_sha256 = digest(0xd1),
            2 => seal.normative_documents.plan_sha256 = digest(0xd2),
            3 => seal.hook_parity_attestation_sha256 = digest(0xd3),
            4 => seal.fixture_bsp_sha256 = digest(0xd4),
            5 => seal.analysis_bsp_sha256 = digest(0xd5),
            6 => seal.executables.cm_sha256 = digest(0xd6),
            7 => seal.executables.pmove_sha256 = digest(0xd7),
            8 => seal.executables.hook_sha256 = digest(0xd8),
            9 => seal.executables.fall_sha256 = digest(0xd9),
            10 => seal.identities.collision.tool_identity = digest(0xda),
            11 => seal.identities.collision.physics_identity = digest(0xdb),
            12 => seal.identities.pmove.tool_identity = digest(0xdc),
            13 => seal.identities.pmove.physics_identity = digest(0xdd),
            14 => seal.identities.hook.tool_identity = digest(0xde),
            15 => seal.identities.hook.physics_identity = digest(0xdf),
            16 => seal.identities.fall.tool_identity = digest(0xe0),
            _ => seal.identities.fall.physics_identity = digest(0xe1),
        }
        assert!(candidate.admit(&bsp).is_err(), "seal mutation {mutation}");
    }

    for mutation in 0..9 {
        let mut candidate = valid.clone();
        match mutation {
            0 => candidate.fall_oracle.tool.name.push('x'),
            1 => candidate.fall_oracle.tool.tool_identity_sha256 = digest(0xe2),
            2 => candidate.fall_oracle.tool.physics_identity_sha256 = digest(0xe3),
            3 => candidate.fall_oracle.parameters.fall_damagemod_f32_bits = 1.5_f32.to_bits(),
            4 => candidate.fall_oracle.parameters.deathmatch = false,
            5 => candidate.fall_oracle.parameters.dmflags = 8,
            6 => candidate.fall_oracle.source.shared_c_sha256 = digest(0xe4),
            7 => candidate.fall_oracle.source.game_header_sha256 = digest(0xe5),
            _ => candidate.fall_oracle.source.build_contract.push('x'),
        }
        candidate.fall_oracle = candidate.fall_oracle.seal();
        assert!(candidate.admit(&bsp).is_err(), "fall mutation {mutation}");
    }
}

#[test]
fn amended_normative_authority_passes_and_superseded_hashes_fail_closed() {
    let bsp = bsp_identity();
    let current = oracle_admissions(&bsp, true, true);
    current.admit(&bsp).unwrap();

    let mut old_design = current.clone();
    old_design
        .b1_runtime_authority_seal
        .normative_documents
        .design_sha256 =
        "eab02d2269f250a26f45bb5d3b1f66ffab2c34ba3ee958d2f8b5bd2a14fef8b5".to_owned();
    assert!(old_design.admit(&bsp).is_err());

    let mut old_plan = current;
    old_plan
        .b1_runtime_authority_seal
        .normative_documents
        .plan_sha256 =
        "970e97b9478b27ad1f1cd35d29a74b2ed2cd51ed1ae8b4af82605615d5b5ba6b".to_owned();
    assert!(old_plan.admit(&bsp).is_err());
}

#[test]
fn hook_requires_companion_pmove_and_canonical_q2ded_parity() {
    let bsp = bsp_identity();
    let valid = oracle_admissions(&bsp, true, true);

    let mut missing_parity = serde_json::to_value(&valid).unwrap();
    missing_parity["hook_oracle"]
        .as_object_mut()
        .unwrap()
        .remove("parity");
    assert!(serde_json::from_value::<OracleAdmissions>(missing_parity).is_err());

    let mut missing_pmove = valid.clone();
    missing_pmove.pmove_oracle = None;
    assert!(missing_pmove.admit(&bsp).is_err());

    for mutation in 0..5 {
        let mut candidate = valid.clone();
        let hook = candidate.hook_oracle.as_mut().unwrap();
        match mutation {
            0 => hook.parity.name.push_str("-wrong"),
            1 => hook.parity.schema.push_str("-wrong"),
            2 => hook.parity.version += 1,
            3 => hook.parity.passed = false,
            _ => hook.parity.case_count -= 1,
        }
        hook.parity = hook.parity.clone().seal();
        *hook = hook.clone().seal();
        assert!(candidate.admit(&bsp).is_err());
    }

    let mut stale_attestation = valid.clone();
    stale_attestation
        .hook_oracle
        .as_mut()
        .unwrap()
        .parity
        .transcript_sha256 = digest(0xba);
    assert!(stale_attestation.admit(&bsp).is_err());

    let mut wrong_tool = valid.clone();
    let hook = wrong_tool.hook_oracle.as_mut().unwrap();
    hook.parity.hook_tool_sha256 = digest(0xbb);
    hook.parity = hook.parity.clone().seal();
    *hook = hook.clone().seal();
    assert!(wrong_tool.admit(&bsp).is_err());

    let mut zero_digest = valid;
    let hook = zero_digest.hook_oracle.as_mut().unwrap();
    hook.parity.q2ded_sha256 = "00".repeat(32);
    hook.parity = hook.parity.clone().seal();
    *hook = hook.clone().seal();
    assert!(zero_digest.admit(&bsp).is_err());
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

    let l0_bytes = u64::from_le_bytes(valid[96..104].try_into().unwrap()) as usize;
    let node_bytes = u64::from_le_bytes(valid[104..112].try_into().unwrap()) as usize;
    let graph_start = 136 + l0_bytes + node_bytes;
    let offset_count =
        u64::from_le_bytes(valid[graph_start..graph_start + 8].try_into().unwrap()) as usize;
    let first_edge = graph_start + 8 + offset_count * 4;
    let mut zero_evidence = valid.clone();
    zero_evidence[first_edge + 20..first_edge + 22].copy_from_slice(&0_u16.to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&zero_evidence, &limits).is_err());
    let mut zero_validation_version = valid.clone();
    zero_validation_version[first_edge + 22..first_edge + 24].copy_from_slice(&0_u16.to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&zero_validation_version, &limits).is_err());

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
    let bsp = bsp_identity();
    let counts = AtlasCounts {
        l0_chunks: 2,
        l1_nodes: 3,
        l1_edges: 2,
        l2_cells: 1,
        l3_cells: 0,
    };
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "fixture.atlas.bin.zst".to_owned(),
        ArtifactManifest::from_uncompressed(
            "application/vnd.q2.atlas-v1",
            payload,
            compressed_size,
            counts.named_counts(),
        ),
    );
    AtlasManifest {
        schema_version: ATLAS_SCHEMA_VERSION,
        byte_order: "little".to_owned(),
        atlas_magic: String::from_utf8_lossy(ATLAS_MAGIC).into_owned(),
        specification_sha256: "ab".repeat(32),
        bsp: bsp.clone(),
        analyzer: identity("q2-atlas", 1),
        oracles: oracle_admissions(&bsp, true, true),
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
        counts,
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
        "cfc926130a3c14de223bccfd0175a57b0c9ceedc5b217b264caf43bcb3b74a60"
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
    assert_eq!(
        restored
            .decode_and_verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &limits)
            .unwrap(),
        artifact(false)
    );

    let mut substitute = artifact(false);
    let a = GridIndex::new(-1, 0, -2);
    let b = GridIndex::new(0, 0, -2);
    let c = GridIndex::new(0, 0, 1);
    let admission = restored.oracles.admit(&restored.bsp).unwrap();
    substitute.l1 = L1Graph::build(
        vec![node(a), node(b), node(c)],
        vec![edge(a, b, EdgeType::StrafeWalk), edge(a, c, EdgeType::Jump)],
        &admission,
        &limits,
    )
    .unwrap();
    assert_eq!(AtlasCounts::from_artifact(&substitute), restored.counts);
    assert!(
        restored
            .verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &substitute, &limits,)
            .is_err()
    );

    let mut lying_artifact_counts = restored.clone();
    lying_artifact_counts
        .artifacts
        .get_mut("fixture.atlas.bin.zst")
        .unwrap()
        .counts
        .insert("l1_edges".to_owned(), 999);
    assert!(
        lying_artifact_counts
            .verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &artifact(false), &limits,)
            .is_err()
    );
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
    bad_digest
        .oracles
        .pmove_oracle
        .as_mut()
        .unwrap()
        .tool
        .executable_sha256 = "ABC".to_owned();
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
fn missing_pmove_authority_is_not_a_canonical_runtime() {
    let limits = AtlasLimits::default();
    let payload = artifact(false).encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&payload, &limits).unwrap();
    let mut collision_only = manifest(&payload, envelope.len() as u64);
    collision_only.oracles.pmove_oracle = None;
    collision_only.oracles.hook_oracle = None;
    assert!(collision_only.canonical_json(&limits).is_err());
    let bytes = manifest(&payload, envelope.len() as u64)
        .canonical_json(&limits)
        .unwrap();
    let mut explicit_null: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    explicit_null["oracles"]["pmove_oracle"] = serde_json::Value::Null;
    assert!(
        AtlasManifest::from_canonical_json(&serde_json::to_vec(&explicit_null).unwrap(), &limits,)
            .is_err()
    );
    let mut explicit_hook_null: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    explicit_hook_null["oracles"]["hook_oracle"] = serde_json::Value::Null;
    assert!(
        AtlasManifest::from_canonical_json(
            &serde_json::to_vec(&explicit_hook_null).unwrap(),
            &limits,
        )
        .is_err()
    );
}

#[test]
fn collision_oracle_is_mandatory_and_manifest_artifact_verification_rechecks_edges() {
    let limits = AtlasLimits::default();
    let decoded_artifact = artifact(false);
    let payload = decoded_artifact.encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&payload, &limits).unwrap();
    let source = manifest(&payload, envelope.len() as u64);
    let bytes = source.canonical_json(&limits).unwrap();
    let mut missing_collision: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    missing_collision["oracles"]
        .as_object_mut()
        .unwrap()
        .remove("collision_oracle");
    assert!(
        AtlasManifest::from_canonical_json(
            &serde_json::to_vec(&missing_collision).unwrap(),
            &limits,
        )
        .is_err()
    );

    let mut no_trajectory = source;
    no_trajectory.oracles.pmove_oracle = None;
    no_trajectory.oracles.hook_oracle = None;
    assert!(no_trajectory.canonical_json(&limits).is_err());
    assert!(
        no_trajectory
            .verify_atlas_artifact(
                "fixture.atlas.bin.zst",
                &payload,
                &decoded_artifact,
                &limits,
            )
            .is_err()
    );
}

fn b3_recovery_graph() -> (AtlasOrigin, L1Graph, GridIndex, GridIndex, GridIndex) {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    let hazard = GridIndex::new(0, 0, 0);
    let east = GridIndex::new(1, 0, 0);
    let north = GridIndex::new(0, 1, 0);
    let mut hazard_node = node(hazard);
    hazard_node.flags &= !NodeFlags::SAFE_TO_STAND;
    hazard_node.hazard_types = PolicyHazardBits::LAVA | PolicyHazardBits::HURT;
    hazard_node.hazard_severity = 192;
    let mut east_edge = edge(hazard, east, EdgeType::Walk);
    east_edge.cost = 512;
    let mut north_edge = edge(hazard, north, EdgeType::StrafeWalk);
    north_edge.cost = 256;
    let mut graph = L1Graph::build(
        vec![node(north), hazard_node, node(east)],
        vec![east_edge, north_edge],
        &admission,
        &limits,
    )
    .unwrap();
    install_static_costs(&mut graph).unwrap();
    install_static_hazard_clearances(&mut graph).unwrap();
    (AtlasOrigin([0, 0, 0]), graph, hazard, east, north)
}

#[test]
fn b3_static_costs_are_exact_and_every_finite_unsafe_node_descends() {
    let (_origin, mut graph, hazard, _east, _north) = b3_recovery_graph();
    let costs = solve_static_costs(&graph).unwrap();
    assert_eq!(costs[graph.node_ordinal(hazard).unwrap()], 256);
    validate_static_costs(&graph).unwrap();

    let mut tampered = costs;
    tampered[graph.node_ordinal(hazard).unwrap()] += 1;
    graph.set_costs_to_safety(&tampered).unwrap();
    assert!(validate_static_costs(&graph).is_err());
}

#[test]
fn b3_signed_hazard_clearance_is_monotonic_deterministic_and_explicit() {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    let indices = (-2..=2)
        .map(|x| GridIndex::new(x, 0, 0))
        .collect::<Vec<_>>();
    let build = |reverse: bool| {
        let mut nodes = indices
            .iter()
            .map(|index| {
                let mut result = node(*index);
                if index.x < 0 {
                    result.flags &= !NodeFlags::SAFE_TO_STAND;
                    result.hazard_types = PolicyHazardBits::LAVA;
                    result.hazard_severity = 255;
                    result.region_id = 11;
                } else {
                    result.region_id = 22;
                }
                result
            })
            .collect::<Vec<_>>();
        let mut edges = Vec::new();
        for pair in indices.windows(2) {
            let mut forward = edge(pair[0], pair[1], EdgeType::Walk);
            forward.cost = 4096;
            let mut backward = edge(pair[1], pair[0], EdgeType::Walk);
            backward.cost = 4096;
            edges.push(forward);
            edges.push(backward);
        }
        if reverse {
            nodes.reverse();
            edges.reverse();
        }
        let mut graph = L1Graph::build(nodes, edges, &admission, &limits).unwrap();
        install_static_costs(&mut graph).unwrap();
        install_static_hazard_clearances(&mut graph).unwrap();
        graph
    };
    let forward = build(false);
    let reverse = build(true);
    let clearances = indices
        .iter()
        .map(|index| forward.nodes()[forward.node_ordinal(*index).unwrap()].hazard_clearance)
        .collect::<Vec<_>>();
    assert_eq!(clearances, vec![-6144, -2048, 2048, 6144, 10240]);
    assert_eq!(
        clearances,
        indices
            .iter()
            .map(|index| {
                reverse.nodes()[reverse.node_ordinal(*index).unwrap()].hazard_clearance
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(
        forward
            .nodes()
            .iter()
            .map(|item| item.region_id)
            .collect::<Vec<_>>(),
        vec![11, 11, 22, 22, 22]
    );
    validate_static_hazard_clearances(&forward).unwrap();
    assert_eq!(
        solve_static_hazard_clearances(&forward).unwrap(),
        clearances
    );

    let raw_forward = AtlasArtifact {
        origin: AtlasOrigin([0, 0, 0]),
        l0: SparseL0::new(),
        l1: forward,
        l2: vec![],
        l3: vec![],
    }
    .encode_uncompressed(&limits)
    .unwrap();
    let raw_reverse = AtlasArtifact {
        origin: AtlasOrigin([0, 0, 0]),
        l0: SparseL0::new(),
        l1: reverse,
        l2: vec![],
        l3: vec![],
    }
    .encode_uncompressed(&limits)
    .unwrap();
    assert_eq!(sha256_hex(&raw_forward), sha256_hex(&raw_reverse));

    let safe_clearance = clearances[2] as f32 / 256.0;
    assert!((0..30).all(|_| safe_clearance >= 0.5));

    let mut isolated_hazard = node(GridIndex::new(0, 0, 0));
    isolated_hazard.flags &= !NodeFlags::SAFE_TO_STAND;
    isolated_hazard.hazard_types = PolicyHazardBits::SLIME;
    let isolated = L1Graph::build(
        vec![isolated_hazard, node(GridIndex::new(2, 0, 0))],
        vec![],
        &admission,
        &limits,
    )
    .unwrap();
    assert_eq!(
        solve_static_hazard_clearances(&isolated).unwrap(),
        vec![
            HAZARD_CLEARANCE_UNREACHABLE_HAZARD,
            HAZARD_CLEARANCE_UNREACHABLE_SAFE
        ]
    );
}

#[test]
fn b3_recovery_packs_primary_alternate_hazards_and_dynamic_repair() {
    assert!(RECOVERY_EVIDENCE_FIELD_NAMES.contains(&"atlas_region_id"));
    assert!(RECOVERY_EVIDENCE_FIELD_NAMES.contains(&"hazard_component_id"));
    assert!(!RECOVERY_EVIDENCE_FIELD_NAMES.contains(&"hazard_region_id"));
    assert!(!RECOVERY_EVIDENCE_FIELD_NAMES.contains(&"server_damage_source_id"));
    let (origin, graph, hazard, east, north) = b3_recovery_graph();
    let position = origin.center(hazard, AtlasLevel::L1);
    let empty = RecoveryOverlay::default();
    let static_block = recovery_features(
        origin,
        &graph,
        RecoveryQuery {
            world_position: position,
            yaw_degrees: 0.0,
            overlay: &empty,
            time_to_impact_seconds: Some(2.0),
        },
    )
    .unwrap();
    assert_eq!(static_block.values.len(), RECOVERY_FEATURE_WIDTH);
    assert_eq!(static_block.values[0], 1.0);
    assert_eq!(static_block.values[2], 1.0);
    assert_eq!(static_block.values[1], 0.0);
    assert_eq!(&static_block.values[9..12], &[0.0, -1.0, 0.0]);
    assert_eq!(&static_block.values[12..15], &[1.0, 0.0, 0.0]);
    assert!((static_block.values[15] - 0.4).abs() < 1e-6);
    assert_eq!(static_block.evidence.l1_index, hazard);
    assert_eq!(static_block.evidence.cost_to_safety_q8, 256);
    assert_eq!(static_block.evidence.signed_safe_clearance_q8, -(8 * 256));
    assert_eq!(
        static_block.evidence.hazard_types,
        PolicyHazardBits::LAVA | PolicyHazardBits::HURT
    );
    assert_eq!(static_block.evidence.atlas_region_id, 7);
    assert_eq!(static_block.evidence.hazard_component_id, 1);

    let mut overlay = RecoveryOverlay::default();
    overlay.blocked_nodes.insert(north);
    let repaired = recovery_features(
        origin,
        &graph,
        RecoveryQuery {
            world_position: position,
            yaw_degrees: 0.0,
            overlay: &overlay,
            time_to_impact_seconds: None,
        },
    )
    .unwrap();
    assert_eq!(&repaired.values[9..12], &[1.0, 0.0, 0.0]);
    assert_eq!(&repaired.values[12..15], &[0.0, 0.0, 0.0]);
    assert_eq!(repaired.repaired_nodes, 3);
    assert_eq!(repaired.evidence.cost_to_safety_q8, 512);
    assert_eq!(
        graph.nodes()[graph.node_ordinal(east).unwrap()].cost_to_safety,
        0
    );
}

#[test]
fn b3_hazard_components_cover_predamage_lava_void_and_safe_recovery_basins() {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    let indices = (-4..=4)
        .map(|x| GridIndex::new(x, 0, 0))
        .collect::<Vec<_>>();
    let build = |reverse: bool| {
        let mut nodes = indices.iter().map(|index| node(*index)).collect::<Vec<_>>();
        let lava = nodes.iter_mut().find(|item| item.index.x == -3).unwrap();
        lava.flags &= !NodeFlags::SAFE_TO_STAND;
        lava.hazard_types = PolicyHazardBits::LAVA;
        lava.hazard_severity = 255;
        let void_edge = nodes.iter_mut().find(|item| item.index.x == 3).unwrap();
        void_edge.flags &= !NodeFlags::SAFE_TO_STAND;
        void_edge.hazard_types = PolicyHazardBits::VOID_OR_LETHAL_DROP;
        void_edge.hazard_severity = 255;
        let mut edges = Vec::new();
        for pair in indices.windows(2) {
            let mut forward = edge(pair[0], pair[1], EdgeType::Walk);
            forward.cost = 4096;
            let mut backward = edge(pair[1], pair[0], EdgeType::Walk);
            backward.cost = 4096;
            edges.extend([forward, backward]);
        }
        if reverse {
            nodes.reverse();
            edges.reverse();
        }
        let mut graph = L1Graph::build(nodes, edges, &admission, &limits).unwrap();
        install_static_costs(&mut graph).unwrap();
        install_static_hazard_clearances(&mut graph).unwrap();
        graph
    };
    let graph = build(false);
    let rebuilt = build(true);
    let components = HazardComponentField::build(&graph).unwrap();
    let rebuilt_components = HazardComponentField::build(&rebuilt).unwrap();
    assert_eq!(components.component_count(), 2);
    assert_eq!(components, rebuilt_components);
    let encode = |l1: L1Graph| {
        AtlasArtifact {
            origin: AtlasOrigin([0, 0, 0]),
            l0: SparseL0::new(),
            l1,
            l2: vec![],
            l3: vec![],
        }
        .encode_uncompressed(&limits)
        .unwrap()
    };
    // The component field is derived from digest-bound canonical graph data;
    // reversing analyzer insertion order changes neither Atlas nor component
    // identity.
    assert_eq!(
        sha256_hex(&encode(graph.clone())),
        sha256_hex(&encode(rebuilt.clone()))
    );

    let id_at = |index: GridIndex| {
        components
            .id_at_ordinal(graph.node_ordinal(index).unwrap())
            .unwrap()
    };
    // Static Atlas bits identify lava entry and exact void/lethal-drop
    // proximity before any server damage-source event exists.
    assert_eq!(id_at(GridIndex::new(-3, 0, 0)), 1);
    assert_eq!(id_at(GridIndex::new(3, 0, 0)), 2);
    // Safe recovery nodes remain associated with the entry component's fixed
    // basin; the distinct void component never aliases traversal region IDs.
    assert_eq!(id_at(GridIndex::new(-2, 0, 0)), 1);
    assert_eq!(id_at(GridIndex::new(-1, 0, 0)), 1);
    assert_eq!(id_at(GridIndex::new(2, 0, 0)), 2);

    let origin = AtlasOrigin([0, 0, 0]);
    let overlay = RecoveryOverlay::default();
    for x in [-3, -2, -1] {
        let block = recovery_features(
            origin,
            &graph,
            RecoveryQuery {
                world_position: origin.center(GridIndex::new(x, 0, 0), AtlasLevel::L1),
                yaw_degrees: 0.0,
                overlay: &overlay,
                time_to_impact_seconds: None,
            },
        )
        .unwrap();
        assert_eq!(block.evidence.hazard_component_id, 1);
        assert_eq!(block.evidence.hazard_component_epoch(17), 17);
        let mut no_component = block.evidence;
        no_component.hazard_component_id = 0;
        assert_eq!(no_component.hazard_component_epoch(17), 0);
        if x >= -2 {
            assert_eq!(block.evidence.cost_to_safety_q8, 0);
            assert!(block.evidence.signed_safe_clearance_q8 > 0);
        }
    }
}

#[test]
fn b3_public_recovery_excludes_hook_and_enables_only_identified_movers() {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    let hazard = GridIndex::new(0, 0, 0);
    let hook_target = GridIndex::new(1, 0, 0);
    let mover_target = GridIndex::new(0, 1, 0);
    let mut hazard_node = node(hazard);
    hazard_node.flags &= !NodeFlags::SAFE_TO_STAND;
    let hook = edge(hazard, hook_target, EdgeType::Hook);
    let mut mover = edge(hazard, mover_target, EdgeType::Mover);
    mover.blocker = 7;
    let mut graph = L1Graph::build(
        vec![hazard_node, node(hook_target), node(mover_target)],
        vec![hook, mover],
        &admission,
        &limits,
    )
    .unwrap();
    install_static_costs(&mut graph).unwrap();
    install_static_hazard_clearances(&mut graph).unwrap();
    assert_eq!(
        graph.nodes()[graph.node_ordinal(hazard).unwrap()].cost_to_safety,
        COST_INFINITY
    );
    let origin = AtlasOrigin([0, 0, 0]);
    let position = origin.center(hazard, AtlasLevel::L1);
    let empty = RecoveryOverlay::default();
    let without_mover = recovery_features(
        origin,
        &graph,
        RecoveryQuery {
            world_position: position,
            yaw_degrees: 0.0,
            overlay: &empty,
            time_to_impact_seconds: None,
        },
    )
    .unwrap();
    assert_eq!(&without_mover.values[9..15], &[0.0; 6]);

    let mut enabled = RecoveryOverlay::default();
    enabled.enabled_mover_blockers.insert(7);
    let with_mover = recovery_features(
        origin,
        &graph,
        RecoveryQuery {
            world_position: position,
            yaw_degrees: 0.0,
            overlay: &enabled,
            time_to_impact_seconds: None,
        },
    )
    .unwrap();
    assert_eq!(&with_mover.values[9..12], &[0.0, -1.0, 0.0]);
    assert_eq!(&with_mover.values[12..15], &[0.0; 3]);
}

#[test]
fn b3_local_repair_refuses_more_than_4096_sparse_nodes() {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, false).admit(&bsp).unwrap();
    let mut nodes = Vec::new();
    'outer: for z in 0..20 {
        for y in 0..20 {
            for x in 0..20 {
                nodes.push(node(GridIndex::new(x, y, z)));
                if nodes.len() == RECOVERY_REPAIR_NODE_LIMIT + 1 {
                    break 'outer;
                }
            }
        }
    }
    let current = GridIndex::new(8, 8, 8);
    assert!(nodes.iter().any(|item| item.index == current));
    let mut graph = L1Graph::build(nodes, vec![], &admission, &limits).unwrap();
    install_static_costs(&mut graph).unwrap();
    install_static_hazard_clearances(&mut graph).unwrap();
    let origin = AtlasOrigin([0, 0, 0]);
    let mut overlay = RecoveryOverlay::default();
    overlay.dynamic_penalty_q8.insert(current, 1);
    let error = recovery_features(
        origin,
        &graph,
        RecoveryQuery {
            world_position: origin.center(current, AtlasLevel::L1),
            yaw_degrees: 0.0,
            overlay: &overlay,
            time_to_impact_seconds: None,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds 4096 L1 nodes"));
}

#[test]
fn b3_guide_slots_are_deterministic_one_hot_and_exactly_60_floats() {
    assert_eq!(ObjectiveClass::Weapon.name(), "weapon");
    assert_eq!(
        ObjectiveClass::from_classname("weapon_rocketlauncher"),
        Some(ObjectiveClass::Weapon)
    );
    assert_eq!(
        ObjectiveClass::from_classname("item_power_shield"),
        Some(ObjectiveClass::Armor)
    );
    assert_eq!(ObjectiveClass::from_classname("light"), None);

    let names = advisory_spatial_feature_names();
    assert_eq!(names.len(), ADVISORY_SPATIAL_WIDTH);
    assert_eq!(
        ADVISORY_SPATIAL_WIDTH,
        RECOVERY_FEATURE_WIDTH + GUIDE_FEATURE_WIDTH
    );
    assert_eq!(ADVISORY_SPATIAL_WIDTH, 76);
    let unique: std::collections::BTreeSet<_> = names.iter().copied().collect();
    assert_eq!(unique.len(), names.len());
}

fn objective_payload(
    bsp_sha256: &str,
    atlas_sha256: &str,
    origin: AtlasOrigin,
    objectives: serde_json::Value,
) -> Vec<u8> {
    let value = serde_json::json!({
        "atlas_sha256": atlas_sha256,
        "bsp_sha256": bsp_sha256,
        "canonical_map_id": "fixture",
        "objectives": objectives,
        "origin": origin.0,
        "schema": "q2-atlas-objectives-v1",
    });
    let mut payload = serde_json::to_vec(&value).unwrap();
    payload.push(b'\n');
    payload
}

fn attach_objective_identity(manifest: &mut AtlasManifest, payload: &[u8], objective_count: u64) {
    manifest.artifacts.insert(
        "fixture.objectives.json".to_owned(),
        ArtifactManifest::from_uncompressed(
            OBJECTIVE_MEDIA_TYPE,
            payload,
            payload.len() as u64,
            BTreeMap::from([("objectives".to_owned(), objective_count)]),
        ),
    );
}

fn fixture_objective_records(origin: AtlasOrigin) -> serde_json::Value {
    let target = GridIndex::new(0, 0, -2);
    let world = origin.center(target, AtlasLevel::L1);
    serde_json::json!([{
        "class": "weapon",
        "classname": "weapon_rocketlauncher",
        "confidence": 65535,
        "l1_index": target.xyz(),
        "objective_id": 7,
        "risk": 0,
        "world_milliunits": world.map(|value| (value * 1000.0).round() as i64),
    }])
}

#[test]
fn b3_runtime_admits_raw_or_enveloped_atlas_and_rejects_identity_mismatch() {
    let limits = AtlasLimits::default();
    let bsp_bytes = b"runtime-bsp-fixture";
    let mut admitted_artifact = artifact(false);
    install_static_costs(&mut admitted_artifact.l1).unwrap();
    install_static_hazard_clearances(&mut admitted_artifact.l1).unwrap();
    admitted_artifact.l2[0].cost_to_safety = 0;
    let raw = admitted_artifact.encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&raw, &limits).unwrap();
    let mut runtime_manifest = manifest(&raw, envelope.len() as u64);
    runtime_manifest.bsp.sha256 = sha256_hex(bsp_bytes);
    runtime_manifest.bsp.size_bytes = bsp_bytes.len() as u64;
    runtime_manifest.oracles = oracle_admissions(&runtime_manifest.bsp, true, true);
    let objectives = objective_payload(
        &runtime_manifest.bsp.sha256,
        &sha256_hex(&raw),
        admitted_artifact.origin,
        serde_json::json!([]),
    );
    let missing_objective_manifest = runtime_manifest.canonical_json(&limits).unwrap();
    assert!(
        AtlasRuntime::from_bytes(
            &missing_objective_manifest,
            "fixture.atlas.bin.zst",
            &envelope,
            "fixture.objectives.json",
            &objectives,
            bsp_bytes,
            "fixture",
            41,
            &limits,
        )
        .is_err()
    );
    attach_objective_identity(&mut runtime_manifest, &objectives, 0);
    let manifest_bytes = runtime_manifest.canonical_json(&limits).unwrap();

    let runtime = AtlasRuntime::from_bytes(
        &manifest_bytes,
        "fixture.atlas.bin.zst",
        &envelope,
        "fixture.objectives.json",
        &objectives,
        bsp_bytes,
        "fixture",
        41,
        &limits,
    )
    .unwrap();
    assert_eq!(runtime.map_epoch(), 41);
    assert_eq!(runtime.atlas_sha256(), sha256_hex(&raw));
    assert_eq!(runtime.shared_instance_count(), 1);
    let clone = runtime.clone();
    assert_eq!(runtime.shared_instance_count(), 2);
    drop(clone);
    assert!(runtime.guide(40, [0.0; 3], 0.0, &[]).is_err());
    let current = runtime.artifact().l1.nodes()[0].index;
    let empty = runtime
        .guide(
            41,
            runtime.artifact().origin.center(current, AtlasLevel::L1),
            0.0,
            &[],
        )
        .unwrap();
    assert_eq!(empty.values, [0.0; GUIDE_FEATURE_WIDTH]);

    let mut wrong_envelope_size = envelope.clone();
    wrong_envelope_size.push(0);
    assert!(
        AtlasRuntime::from_bytes(
            &manifest_bytes,
            "fixture.atlas.bin.zst",
            &wrong_envelope_size,
            "fixture.objectives.json",
            &objectives,
            bsp_bytes,
            "fixture",
            41,
            &limits,
        )
        .is_err()
    );

    assert!(
        AtlasRuntime::from_bytes(
            &manifest_bytes,
            "fixture.atlas.bin.zst",
            &raw,
            "fixture.objectives.json",
            &objectives,
            bsp_bytes,
            "wrong-map",
            41,
            &limits,
        )
        .is_err()
    );
    assert!(
        AtlasRuntime::from_bytes(
            &manifest_bytes,
            "fixture.atlas.bin.zst",
            &raw,
            "fixture.objectives.json",
            &objectives,
            b"wrong-bsp",
            "fixture",
            41,
            &limits,
        )
        .is_err()
    );
}

fn b3_runtime_fixture(map_epoch: u64) -> AtlasRuntime {
    b3_runtime_fixture_with_extra_limit(map_epoch, None)
}

fn b3_runtime_fixture_with_extra_limit(map_epoch: u64, extra_limit: Option<&str>) -> AtlasRuntime {
    let limits = AtlasLimits::default();
    let bsp_bytes = b"runtime-bsp-fixture";
    let mut admitted_artifact = artifact(false);
    install_static_costs(&mut admitted_artifact.l1).unwrap();
    install_static_hazard_clearances(&mut admitted_artifact.l1).unwrap();
    admitted_artifact.l2[0].cost_to_safety = 0;
    let raw = admitted_artifact.encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&raw, &limits).unwrap();
    let mut runtime_manifest = manifest(&raw, envelope.len() as u64);
    runtime_manifest.bsp.sha256 = sha256_hex(bsp_bytes);
    runtime_manifest.bsp.size_bytes = bsp_bytes.len() as u64;
    runtime_manifest.oracles = oracle_admissions(&runtime_manifest.bsp, true, true);
    let objectives = objective_payload(
        &runtime_manifest.bsp.sha256,
        &sha256_hex(&raw),
        admitted_artifact.origin,
        fixture_objective_records(admitted_artifact.origin),
    );
    attach_objective_identity(&mut runtime_manifest, &objectives, 1);
    if let Some(extra_limit) = extra_limit {
        runtime_manifest.limitations.push(extra_limit.to_owned());
    }
    let manifest_bytes = runtime_manifest.canonical_json(&limits).unwrap();
    AtlasRuntime::from_bytes(
        &manifest_bytes,
        "fixture.atlas.bin.zst",
        &envelope,
        "fixture.objectives.json",
        &objectives,
        bsp_bytes,
        "fixture",
        map_epoch,
        &limits,
    )
    .unwrap()
}

#[test]
fn b3_runtime_slot_is_epoch_atomic_and_query_components_are_counted() {
    let slot = AtlasRuntimeSlot::new();
    assert!(slot.snapshot(41).is_err());

    slot.activate(b3_runtime_fixture(41)).unwrap();
    let retained = slot.snapshot(41).unwrap();
    // Re-admitting the same immutable identity in one epoch is idempotent.
    slot.activate(b3_runtime_fixture(41)).unwrap();
    assert!(
        slot.activate(b3_runtime_fixture_with_extra_limit(
            41,
            Some("different manifest identity")
        ))
        .is_err()
    );

    let current = retained.artifact().l1.nodes()[0].index;
    let origin = retained.artifact().origin;
    let overlay = RecoveryOverlay::default();
    let timed = retained
        .advisory_spatial_features_timed(
            41,
            RecoveryQuery {
                world_position: origin.center(current, AtlasLevel::L1),
                yaw_degrees: 0.0,
                overlay: &overlay,
                time_to_impact_seconds: None,
            },
            &[],
        )
        .unwrap();
    assert_eq!(timed.values.len(), ADVISORY_SPATIAL_WIDTH);
    assert!(timed.timing.total_ns >= timed.timing.atlas_lookup_ns);
    assert!(timed.timing.total_ns >= timed.timing.recovery_ns);
    assert!(timed.timing.total_ns >= timed.timing.guide_ns);
    let counters = retained.query_counters();
    assert_eq!(counters.accepted_queries, 1);
    assert_eq!(counters.atlas_lookup_ns, timed.timing.atlas_lookup_ns);
    assert_eq!(counters.recovery_ns, timed.timing.recovery_ns);
    assert_eq!(counters.guide_ns, timed.timing.guide_ns);
    assert_eq!(counters.total_ns, timed.timing.total_ns);
    let guide = retained
        .guide(
            41,
            origin.center(current, AtlasLevel::L1),
            0.0,
            &[ObjectiveBelief {
                objective_id: 7,
                availability_belief: 0.75,
            }],
        )
        .unwrap();
    assert!(guide.values[..3].iter().any(|value| *value != 0.0));
    assert!((guide.values[6] - 0.75).abs() < 1e-5);
    assert_eq!(guide.values[7 + ObjectiveClass::Weapon as usize], 1.0);

    let epoch_41_evidence = retained
        .recovery(
            41,
            RecoveryQuery {
                world_position: origin.center(current, AtlasLevel::L1),
                yaw_degrees: 0.0,
                overlay: &overlay,
                time_to_impact_seconds: None,
            },
        )
        .unwrap()
        .evidence;

    slot.activate(b3_runtime_fixture(42)).unwrap();
    let epoch_42 = slot.snapshot(42).unwrap();
    assert_eq!(epoch_42.map_epoch(), 42);
    let epoch_42_evidence = epoch_42
        .recovery(
            42,
            RecoveryQuery {
                world_position: origin.center(current, AtlasLevel::L1),
                yaw_degrees: 0.0,
                overlay: &overlay,
                time_to_impact_seconds: None,
            },
        )
        .unwrap()
        .evidence;
    assert_eq!(
        epoch_41_evidence.hazard_component_id,
        epoch_42_evidence.hazard_component_id
    );
    assert_eq!(retained.atlas_sha256(), epoch_42.atlas_sha256());
    assert!(slot.snapshot(41).is_err());
    // A reader holding the prior snapshot remains isolated from activation.
    assert_eq!(retained.map_epoch(), 41);
    assert!(slot.activate(b3_runtime_fixture(40)).is_err());
    assert!(slot.clear(41).is_err());
    slot.clear(42).unwrap();
    assert!(slot.snapshot(42).is_err());
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
