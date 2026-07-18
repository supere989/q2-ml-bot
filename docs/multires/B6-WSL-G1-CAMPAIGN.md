# B6 WSL G1 No-Update Campaign

B6 is a distinct pre-training transport gate. It collects exactly 16,384
accepted transitions on `DESKTOP-RTX2080` WSL2 with the step-zero attested
checkpoint and performs no policy, optimizer, backward, or checkpoint update.
It is not the B7 campaign: B7 must re-prove G1 and then G2 after updates exist.

No B6 producer promotes a trainer, modifies the public server, writes a map
queue, or starts TensorBoard. Every output path is exclusive; an existing raw
or aggregate evidence file is a hard refusal.

## Evidence method

The green conclusion is derived from four independent evidence families:

1. `train/multires_one_run.py --campaign_mode b6-wsl-g1-no-update` emits the
   hashed raw four-client trajectory, authoritative echo facts, water/land
   vertical projection observations, loaded checkpoint/policy/optimizer
   before-and-after identities, update counters, PID/start-tick ownership, six
   actual UDP bind records, and four separate protocol qport identities.
2. `tools/qualify_network_client_frame_barrier.py run` executes the real
   full-network fault campaign. Its execution seal now binds the launch host
   and machine identity. `tools/assemble_b6_g1_fault_probe.py` revalidates all
   hashed raw scenarios and derives map-epoch recovery, whole-cohort telemetry
   gap recovery, and fatal partial-client timeout from those raw observations.
   The gap is injected by a bounded UDP relay which holds the first real
   telemetry datagram, suppresses later telemetry while held, and releases the
   identical first bytes after the collector timeout but before the server
   timeout. Relay endpoints, byte hashes, timing, suppression counts, and live
   q2ded/client process identities are raw evidence. It does not accept a
   receive-method monkeypatch, a command-triple drop, or handwritten pass
   booleans as transport proof.
3. `tools/run_b6_attested_campaign.py` owns the campaign order and the fresh
   256-bit run nonce. It invokes an Ed25519-signed public pre-probe, derives the
   WSL `launch_id` from the nonce and signed pre-probe evidence digest, runs the
   exact no-update collection once, and invokes a signed public post-probe
   whose predecessor is the one-run evidence digest. Its sealed ledger records
   every exact argv and the three local monotonic intervals; no cross-host wall
   clock comparison is used.
4. `tools/capture_b6_public_state.py` authenticates the exact public and teacher
   services against `B6-PUBLIC-HOST-AUTHORITY.json`. Unit/drop-in bytes, exact
   ExecStart and working directory, running executable, PID/start-tick,
   monotonic activation timestamps, socket owner/inode, delivered-map trees,
   runtime artifacts, and credential metadata are compared exactly. All path
   components are no-follow and files use stat/read/stat identity checks.
   Credential contents are never emitted; only a domain-separated opaque HMAC
   derived from the dedicated signing key is recorded.
5. The B2 WSL Dyn evidence supplies measured four-client p99, Atlas/Dyn
   resident sizes, build RSS, and the same canonical machine identity. B3
   supplies the exact hook-necessity identity: 15 walking ticks, 10 Hz game
   cadence, and 76,800 q8 walking units per second.

`tools/assemble_b6_wsl_g1_campaign.py` then cross-binds those observations to
the exact B3/B4/B5 gates, runtime manifest, training manifest, objectives,
bundle, Atlas manifest/raw Atlas, attested checkpoint, lineage evidence, and
retirement evidence. Optimizer authority is the optimizer loaded from the
attested checkpoint and hashed by the one-run producer; there is no unrelated
external optimizer-state file.

## Required order (single controller)

Use a new evidence directory and absolute paths throughout. The following is
an operational template, not an instruction to launch automatically.

1. Provision the dedicated public-host Ed25519 key once with
   `tools/provision_b6_public_probe_key.py`. The private PEM must remain at the
   authority-pinned absolute path with mode `0600`; pin only its activation
   record's public key and key ID. Provisioning refuses to overwrite either
   destination. Copy the exact capture tool and canonical authority to the
   evidence-only public-host directory without changing either live service.

2. On `DESKTOP-RTX2080` WSL2, produce a fresh full-network execution with the
   exact B6 q2ded/game/client bytes and finalize its normal runtime
   qualification. The run command already executes scenarios concurrently via
   its fixed job pool. Reusing an execution produced before launch-host
   provenance existed is forbidden.

3. Still on WSL2, assemble the G1 fault probe from that execution:

   ```sh
   python3 tools/assemble_b6_g1_fault_probe.py \
     --execution <absolute-frame-barrier-execution.json> \
     --runtime-manifest <runtime-manifest.json> \
     --training-manifest <training.json> --objectives <map.objectives.json> \
     --checkpoint <fresh-step-zero.pt> --bundle-manifest <map.bundle.json> \
     --atlas <map.atlas.bin> --atlas-manifest <map.atlas.manifest.json> \
     --b3-gate <B3-GATE.json> --output <absolute-new-fault-probe.json>
   ```

4. Create one canonical controller plan. Its SSH argv, remote Python/tool/
   authority/key/output paths, and one-run argv are exact. The one-run argv
   must select `b6-wsl-g1-no-update` but must not contain `--launch_id` or
   `--out`; those are controller-owned. Seal it as
   `q2-multires-b6-attested-campaign-plan-v1`.

5. From WSL2, run the controller once:

   ```sh
   python3 tools/run_b6_attested_campaign.py \
     --plan <absolute-controller-plan.json> \
     --output-root <absolute-new-controller-evidence-directory>
   ```

   This is the only admitted way to order the public pre-probe, exact four-client
   16,384-transition no-update run, and public post-probe. The pre signature is
   chained from the authority seal; the one-run launch ID is chained from the
   signed pre evidence; the post signature is chained from the one-run
   evidence. Any command failure, signature failure, replayed nonce/phase,
   capture-tool drift, service restart, socket change, map/runtime/credential
   change, host change, or missing service is fatal.

6. Assemble one canonical aggregate using
   `tools/assemble_b6_wsl_g1_campaign.py`. Supply both the summarized fault
   probe and its exact raw full-network execution with `--fault-execution`;
   the assembler revalidates every referenced raw scenario before accepting
   the summary. Also supply the one-run, signed public probes, controller
   plan/ledger, B2/B3/B4/B5 evidence,
   lineage/retirement evidence, and every exact runtime artifact requested by
   `--help`. In particular, `--runtime-evidence` is the compact
   `B4/b4-wire-generation.json` already bound by B5, while
   `--runtime-manifest` is the full sealed runtime manifest already bound by
   B4 and loaded by one-run. The assembler requires both byte identities and
   requires their semantic `runtime_manifest_sha256` values to agree. It then
   independently rederives the invocation/hash chain and monotonic order.
   Place that aggregate in the `wsl_b6_campaign` slot of
   `tools/verify_multires_integration.py`.

## Green thresholds

- Exactly 16,384 accepted raw transitions; failed rounds and echo timeouts are
  zero.
- Authoritative echo acceptance is at least 0.97.
- Vertical enum/upmove agreement is at least 0.99 over accepted transitions
  outside bounded declared resyncs.
- Both water and land are sampled; both projection mismatch counts and their
  skew are zero.
- Map-epoch recovery and whole-batch telemetry-gap recovery are exercised;
  one live-client telemetry timeout is fatal. Command-triple brief-drop and
  process-disconnect/SIGKILL remain distinct tests and cannot satisfy either
  predicate.
- The declared resync limit is capped at 64. Accepted-frame discontinuities
  must be strictly forward and cannot outnumber sealed boundary admissions;
  repeated non-trainable whole-batch gaps fail closed at the collector cap.
- Four-client feature assembly p99 is positive and below 0.5 ms; Atlas is at
  most 32 MiB, four Dyn snapshots are below 8 MiB, and Atlas build peak RSS is
  at most 512 MiB.
- LAN payload is accounted from the imported wire ABI (`struct.calcsize`): the
  client datagram closes as header + engine observation + causal telemetry,
  and the action datagram closes from its action struct. Atlas and Dyn have
  zero wire fields and therefore zero per-frame wire bytes.
- Checkpoint, policy state, and loaded optimizer state are byte-identical by
  canonical identity before and after; all update counters are zero.
- Public pre/post state is identical and all scoped staging processes and real
  UDP binds are gone after teardown.

No green B6 result authorizes primary-trainer cutover by itself. The exact
post-B6 envelope/report, cold-start bindings, source freeze, and two-pass
trainer/TensorBoard launch gate are normative in
[`PRIMARY-TRAINER-ADMISSION.md`](PRIMARY-TRAINER-ADMISSION.md).
