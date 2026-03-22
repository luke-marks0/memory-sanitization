# PoSE Cutover Checklist

This checklist tracks the repository's hard cutover from the old Filecoin PoRep-based design to the graph-based PoSE-DB design required by `docs/repository-spec.md`.

Cutover rule:
- [x] Do not preserve runtime, schema, test, benchmark, or documentation compatibility with PoRep.
- [x] Do not retain PoRep codepaths as legacy or fallback behavior.
- [x] Treat any remaining PoRep references in supported code or docs as migration bugs until removed.

## 1. Repository And Documentation Alignment

- [x] Move the bundled paper to `docs/references/software-based-memory-erasure-relaxed-isolation.pdf`.
- [x] Remove or replace the old top-level `repository-spec.md` so the repo has one authoritative spec story.
- [x] Add `docs/graph-construction.md`.
- [x] Add `docs/security-model.md`.
- [x] Rewrite `docs/architecture.md` for one protocol with initialization plus fast phase, not inner/outer proofs.
- [x] Rewrite `docs/protocol.md` around `Discover`, `PlanSession`, `LeaseRegions`, `SeedSession`, `MaterializeLabels`, `PrepareFastPhase`, `RunFastPhase`, `Finalize`, and `Cleanup`.
- [x] Rewrite `docs/result-schema.md` to the PoSE-DB result artifact fields.
- [x] Rewrite `docs/benchmarking.md` to include calibration artifacts, `q`, `gamma`, attacker-budget reporting, and rechallenge semantics.
- [x] Rewrite `docs/threat-model.md` around the distant-attacker model, random-oracle assumption, and honest `M` accounting.
- [x] Delete PoRep-only documents that no longer describe supported behavior.
- [x] Remove PoRep language from `README.md`.
- [x] Rename package/project descriptions in `pyproject.toml`, `Cargo.toml`, and any remaining metadata from Filecoin/PoRep wording to PoSE/PoSE-DB wording.

## 2. Required Repository Layout

- [x] Add `src/pose/graphs/`.
- [x] Add `src/pose/hashing/`.
- [x] Add `src/pose/cli/calibrate.py`.
- [x] Add `src/pose/benchmarks/calibration.py`.
- [x] Add any new native-acceleration directories only if they are parity-gated against Python reference semantics.
- [x] Update layout tests to enforce the new spec-required directories and files.
- [x] Remove layout tests that require `vendor/rust-fil-proofs` or `rust/pose_filecoin_bridge`.

## 3. Common Utilities And Canonical Encodings

- [x] Add a shared deterministic CBOR helper in `src/pose/common/cbor.py`.
- [x] Move canonical CBOR usage out of PoRep-specific modules into common or PoSE-DB modules.
- [x] Add domain-separated hash input encoding helpers for graph descriptors and labels.
- [x] Add hash backend selection plumbing for `blake3-xof` default and optional `shake256`.
- [x] Replace PoRep-specific timing keys in `src/pose/common/timing.py` with PoSE-DB timing keys.
- [x] Update integrity/shortcut scanning to ban PoSE-DB-specific shortcut patterns rather than PoRep-specific wording only.

## 4. Python Reference PoSE-DB Core

- [x] Implement the `pose-db-drg-v1` graph family in `src/pose/graphs/`.
- [x] Implement arbitrary-`m` graph construction and selection of the smallest `n` with `2^(n+1) >= m`.
- [x] Implement deterministic topological node numbering.
- [x] Implement deterministic predecessor ordering.
- [x] Implement deterministic ordered challenge set `O(G)`.
- [x] Implement graph descriptor encoding in deterministic CBOR.
- [x] Implement `graph_descriptor_digest`.
- [x] Implement canonical label semantics for source nodes and internal nodes.
- [x] Implement `w_bits` and `w_bytes` validation rules.
- [x] Implement reference challenge-set indexing by slot position.
- [x] Add exhaustive small-case graph/property tests.
- [x] Add deterministic reference vectors for graph descriptors, node orderings, predecessors, and labels.

## 5. Soundness And Calibration

- [x] Implement verifier-side soundness calculations for graph-restricted and general models.
- [x] Implement conservative rounding in the verifier's favour.
- [x] Reject invalid profiles where `w0 <= 0`.
- [x] Reject invalid profiles where `M' / m >= 1`.
- [x] Reject invalid profiles where the additive term already exceeds the target bound.
- [x] Implement `q` derivation from measured fastest local hash throughput, transport overhead, resident lookup latency, and safety margin.
- [x] Enforce `q < gamma` before allowing a production profile to run.
- [x] Persist calibration artifacts per profile.
- [x] Add CLI support for explicit calibration runs.
- [x] Add tests that fail when calibration is invalid or attacker-budget accounting is inconsistent.

## 6. Session Planning And Region Layout

- [x] Replace PoRep-unit planning with slot-based planning.
- [x] Update benchmark profiles to define `w_bits`, graph family, hash backend, adversary model, attacker-budget assumption, target bound or fixed `r`, deadline policy, calibration policy, and cleanup policy.
- [x] Compute per-region `slot_count = floor(usable_region_bytes / w_bytes)`.
- [x] Compute global `m` from all challenged regions.
- [x] Compute `gamma` from the selected `n`.
- [x] Compute `covered_bytes`, `slack_bytes`, and `coverage_fraction`.
- [x] Define canonical global region order and in-region slot ordering.
- [x] Record claim notes whenever unchallenged local tiers materially affect `M`.
- [x] Enforce coverage threshold handling using PoSE-DB covered bytes rather than `real_porep_ratio`.
- [x] Rewrite host planning around host slot counts.
- [ ] Rewrite GPU planning around GPU slot counts.
- [ ] Add hybrid planning across host plus HBM.
- [ ] Add 8xH100 aggregation planning.

## 7. Protocol And Schemas

- [x] Rewrite `proto/pose/v1/session.proto` for the PoSE-DB protocol.
- [x] Regenerate Python protobuf bindings.
- [x] Rewrite `src/pose/protocol/messages.py`.
- [x] Rewrite `src/pose/protocol/grpc_codec.py`.
- [x] Rewrite `src/pose/protocol/result_schema.py`.
- [x] Replace `GenerateInnerPoRep` with `SeedSession`.
- [x] Replace `MaterializeRegionPayloads` with `MaterializeLabels`.
- [x] Replace `CommitRegions`, `VerifyInnerProofs`, `ChallengeOuter`, and `VerifyOuter` with `PrepareFastPhase` and `RunFastPhase`.
- [x] Keep `Discover`, `PlanSession`, `LeaseRegions`, `Finalize`, and `Cleanup` only if their semantics match the new spec.
- [x] Remove `porep_unit_profile`, `inner_proof_mode`, `sector_plan`, `real_porep_*`, `inner_filecoin_verified`, `outer_pose_verified`, and Merkle-specific payload fields from supported schemas.
- [x] Update retained-session and rechallenge state formats to the new slot-based fast phase.

## 8. Prover Implementation

- [x] Replace PoRep-based prover state with session seed plus graph/session parameters.
- [x] Implement implicit graph construction in the prover.
- [x] Implement in-place label generation into leased host regions.
- [ ] Implement in-place label generation into leased HBM regions.
- [x] Measure and report bounded scratch usage separately from covered state.
- [x] Enforce stage-buffer cleanup or declaration before the fast phase.
- [x] Implement fast-round responses by direct slot lookup, not Merkle openings.
- [x] Ensure future challenges remain hidden from the prover before each round.
- [x] Rewrite resident-session behavior for PoSE-DB rechallenge.
- [x] Remove all prover imports from `src/pose/filecoin/` and any Filecoin bridge APIs.

## 9. Verifier Implementation

- [x] Replace Merkle/outer-proof verification with direct label verification for challenged slot indices.
- [x] Sample one challenge index per round uniformly from `[0, m)` with replacement.
- [x] Start timing immediately before challenge emission.
- [x] Enforce per-round deadline `Delta`.
- [x] Precompute the challenge schedule when allowed by the spec.
- [x] Precompute expected labels for the sampled schedule on the verifier side.
- [x] Verify every round's response bytes against the expected label.
- [x] Produce final verdicts using the new spec's verdict set.
- [x] Produce formal-claim and operational-claim notes separately.
- [x] Rewrite host verifier sessions around PoSE-DB semantics.
- [ ] Rewrite GPU verifier sessions around PoSE-DB semantics.
- [ ] Implement hybrid verifier sessions.
- [ ] Implement 8xH100 verifier orchestration.

## 10. Memory Backends And Fast-Phase Transport

- [x] Keep verifier-owned host leases and adapt them for slot-based reads and writes.
- [ ] Keep verifier-owned CUDA IPC leases and adapt them for slot-based reads and writes.
- [ ] Ensure production HBM mode does not use managed/unified memory.
- [x] Add explicit reporting for any host shadow or stage copy that survives into the fast phase.
- [x] Confirm fast-phase transport is low-jitter enough for calibrated profiles.
- [x] If gRPC remains on the fast path, ensure calibration captures its actual overhead and jitter.
- [ ] Add a lower-overhead fast-phase transport if gRPC cannot support valid production calibration margins.

## 11. CLI, Benchmarking, And Artifacts

- [x] Rewrite `pose verifier run` output around `m`, `gamma`, `r`, `Delta`, `q`, attacker budget, and reported success bound.
- [x] Add `pose verifier calibrate --profile ...`.
- [x] Update `pose verifier rechallenge` to the PoSE-DB resident-label model.
- [x] Update `pose bench run`, `pose bench matrix`, and `pose bench summarize` for calibration-aware PoSE-DB metrics.
- [x] Rewrite benchmark summaries to report `q`, `gamma`, coverage, slack, attacker-budget assumption, and soundness bound.
- [x] Ensure result artifacts include all fields required by the new spec.
- [x] Ensure human-readable summaries do not overstate tier-specific claims.
- [x] Update `Makefile` targets to match the new build, calibration, and test workflows.

## 12. Tests

- [x] Replace PoRep serialization tests with graph descriptor and label semantics tests.
- [x] Replace real-bridge e2e tests with PoSE-DB end-to-end host sessions.
- [x] Add paper-conformance tests for graph family, arbitrary-`m`, challenge distribution, and soundness formulas.
- [x] Add parity tests between Python reference code and any native accelerators.
- [x] Add graph property tests for small exhaustive cases and large structural invariants.
- [x] Add label-correctness tests for wrong seed, wrong backend, wrong ordering, and stale replay.
- [x] Add protocol tests for malformed messages, duplicate session IDs, timeout handling, and fast-phase transport reporting.
- [x] Add in-place and staging tests that reject hidden `O(m)` shadow state.
- [x] Add adversarial tests for recomputation-on-demand, hidden host shadows, sparse writes, copy-in after challenge, and bad `M` accounting on the host-only path.
- [ ] Add adversarial tests for hidden HBM shadows.
- [x] Update hardware test scaffolding for single-H100 host-only profiles and documentation.
- [ ] Update hardware tests for single-H100 HBM-only, single-H100 hybrid, 8xH100 HBM-only, and 8xH100 hybrid.
- [x] Update performance tests to track label generation time, HBM transfer time, fast-phase latency, `q/gamma` margin, coverage fraction, and total runtime.
- [x] Remove tests that require vendored Filecoin code, the bridge, PoRep witness units, or ex-post C2.

## 13. Code And Asset Deletion

- [x] Delete `src/pose/filecoin/`.
- [x] Delete `rust/pose_filecoin_bridge/`.
- [x] Delete `vendor/rust-fil-proofs/`.
- [x] Delete Filecoin upstream sync and verification scripts.
- [x] Delete Filecoin bridge build scripts.
- [x] Delete PoRep fixtures and generated vectors that are no longer relevant.
- [x] Delete PoRep-specific benchmark profiles, including ex-post variants.
- [x] Delete PoRep-specific docs such as serialization and ex-post C2 material if they no longer describe supported behavior.
- [x] Delete PoRep-specific result fields, plan fields, and session-store fields.
- [x] Delete any remaining references to Filecoin or PoRep from supported CLI help text.

## 14. Final Cutover Validation

- [x] `make build` succeeds on the new codebase.
- [x] `make test` succeeds without PoRep dependencies or tooling.
- [x] `make test-parity` succeeds for the Python reference and any enabled accelerators.
- [x] `make test-graphs` exists and succeeds.
- [x] `make calibrate PROFILE=dev-small` succeeds.
- [x] `make bench PROFILE=dev-small` succeeds.
- [x] `dev-small` runs an end-to-end host PoSE-DB session successfully.
- [ ] `single-h100-hbm-max` runs an end-to-end HBM PoSE-DB session successfully.
- [ ] `single-h100-hybrid-max` runs an end-to-end hybrid PoSE-DB session successfully.
- [ ] `eight-h100-hbm-max` runs an end-to-end 8xH100 HBM PoSE-DB session successfully.
- [ ] `eight-h100-hybrid-max` runs an end-to-end 8xH100 hybrid PoSE-DB session successfully.
- [ ] All production profiles record valid `q < gamma`.
- [x] Result artifacts include verdict, coverage, attacker-budget assumption, soundness model, and timings.
- [x] No supported-path code or docs mention PoRep, Filecoin, ex-post C2, inner proof, or outer proof except as removed-history context outside the supported repository surface.

## 15. Recommended Execution Order

- [ ] Complete repository/doc alignment and required layout changes first.
- [ ] Land the Python reference graph, hashing, and soundness core second.
- [ ] Replace schemas and protocol types third.
- [ ] Rebuild host planning and host sessions fourth.
- [ ] Add calibration and reporting fifth.
- [ ] Rebuild GPU, hybrid, and 8xH100 paths sixth.
- [x] Remove PoRep code and assets only after the new host path is green.
- [ ] Close the migration only after all final validation boxes are checked.
