# Repository Specification: Python PoSE Using Real Filecoin PoRep

**Status:** Draft v1  
**Audience:** Maintainers and contributors implementing the repository  
**Normative language:** “must”, “must not”, “required”, “shall”, and “should” are normative unless a section is explicitly marked informative.

---

## 1. Purpose

This repository shall implement a **real cryptographic proof of secure erasure (PoSE)** over **user-space accessible host memory and GPU HBM** by forcing a prover process to overwrite challenged memory regions with a **genuine Filecoin PoRep-derived object** and then prove possession of that object under verifier challenge.

The repository is **Python-first** in the orchestration layer and **Rust-reference-backed** in the cryptographic layer:

- Python owns the CLI, prover process, verifier process, protocol, region planning, memory allocators, benchmarking, result reporting, and the staged Filecoin port.
- Vendored Filecoin Rust code owns the production inner PoRep path until an explicit parity-gated promotion moves a component into the Python port.

The top priority is correctness and cryptographic credibility:

1. The inner proof must be a real Filecoin PoRep produced by the official Rust implementation.
2. The outer proof must prove storage of the concrete PoRep object resident in the challenged host/HBM region.
3. The repository must keep prover and verifier concerns cleanly separated so that the PoSE can be benchmarked, stressed, and iterated on without compromising the security story.

---

## 2. Goals

### 2.1 Primary goals

The repository must:

- implement a **real Filecoin PoRep** path using vendored upstream Rust code;
- expose that path through a thin Rust bridge to Python;
- implement a **clean prover/verifier split** as separate processes;
- implement PoSE for:
  - host memory only,
  - GPU HBM only,
  - hybrid host + HBM sessions;
- support a CLI that runs the erasure protocol and outputs:
  - whether the PoSE succeeded,
  - how much memory was covered,
  - how long each stage took;
- maximize coverage of **user-space accessible** memory on:
  - an initial single-H100 machine,
  - a later 8×H100 machine;
- include a comprehensive test suite that proves:
  - the inner Filecoin functionality is real,
  - the Python mirror is identical where required,
  - the PoSE protocol behaves correctly and fails safely.

### 2.2 Secondary goals

The repository should:

- support local benchmarking and repeated rechallenge runs;
- make it easy to compare host-only, HBM-only, and hybrid PoSE profiles;
- allow staged migration of deterministic Filecoin subcomponents into Python;
- preserve a strong audit trail back to upstream Filecoin code.

---

## 3. Non-goals

The repository does **not** claim to solve or provide:

- kernel-level or firmware-level memory erasure;
- attestation of inaccessible memory or privileged memory;
- erasure guarantees for disk, SSD, NVMe, or remote storage;
- protection against a malicious OS kernel, malicious NVIDIA driver, or malicious device firmware;
- production blockchain integration;
- “fake”, “synthetic”, or shortcut Filecoin proofs in production mode.

The repository’s security claims are limited to the declared protocol, the challenged user-space memory regions, and the trusted-computing-base assumptions documented below.

---

## 4. Core security statement

A PoSE session is considered **successful** only if all of the following hold:

1. The prover produced one or more **real Filecoin PoRep sector proofs** using the vendored upstream Rust pipeline.
2. The verifier accepted those sector proofs using the official verification path.
3. The prover serialized a canonical **PoRep object** into the challenged host/HBM region(s).
4. The verifier accepted a time-bounded outer proof of storage over those serialized bytes.
5. The challenged object occupied at least the configured fraction of the challenged user-space region budget.
6. The prover satisfied the verifier’s challenge deadline.

A successful PoSE therefore means:

- the prover demonstrated possession of a **real Filecoin PoRep artifact**; and
- the prover demonstrated timely possession of the bytes currently committed in the challenged memory region.

This repository shall implement that security claim as the conjunction of:

- an **inner proof**: Filecoin PoRep; and
- an **outer proof**: timed proof of storage over the in-memory serialized PoRep object.

Neither proof alone is sufficient.

---

## 5. Memory allocation ownership

The verifier process owns the challenged memory allocations and leases them to the prover process:

- for host memory: shared anonymous mappings, `memfd`, or equivalent verifier-owned shared mappings;
- for HBM: verifier-owned CUDA allocations exported to the prover by CUDA IPC handles or an equivalent same-host device-sharing mechanism.

The verifier chooses the exact challenged user-space regions, can account for their size, can track lifecycle, and can enforce cleanup semantics.

---

## 6. Threat model and assumptions

### 6.1 Attacker model

The prover process may be adversarial and may attempt to:

- reuse stale but valid old Filecoin proofs;
- prove storage of different bytes than the genuine PoRep object;
- keep the object outside the challenged region and copy only challenged leaves on demand;
- partially overwrite a region rather than fully filling it;
- exploit protocol ambiguity, replay, race conditions, or stale manifests;
- respond after a delay that hides reconstruction or fetch costs.

### 6.2 Trusted computing base

The trusted computing base includes:

- the verifier process;
- the operating system kernel;
- the file/memory sharing primitives used for leased regions;
- the Rust bridge and vendored Filecoin Rust code;
- the Python verifier implementation;
- the CUDA runtime and NVIDIA driver in HBM mode;
- the cryptographic hash used for the outer proof.

### 6.3 Timing assumption

The outer PoSE is time-bounded. The verifier’s deadline assumes:

- proving possession of bytes already resident in the challenged region is faster than:
  - reconstructing the PoRep object,
  - fetching it from elsewhere,
  - or moving it from a non-challenged location into the challenged region after challenge.

The protocol shall therefore maintain separate benchmark data for:

- resident-response time;
- copy-from-alternate-store time;
- rebuild-or-rematerialize time.

### 6.4 Coverage claim

The repository may only claim coverage for:

- the challenged leased/declared host-memory regions;
- the challenged leased/declared HBM regions.

It must never report “box memory erased” or “full GPU erased” unless the reported coverage explicitly equals the measured user-space accessible capacity under the active benchmark profile.

---

## 7. Terminology

- **PoRep unit**: one real Filecoin sealing instance and its associated serialized artifacts.
- **Region**: one challenged memory allocation, either host or HBM.
- **Region payload**: the raw bytes written into a challenged region.
- **Session**: one verifier challenge and the prover’s full response.
- **Coverage bytes**: bytes in challenged regions whose contents are committed by the outer proof.
- **Real PoRep bytes**: bytes in the payload that correspond to genuine Filecoin PoRep artifacts.
- **Tail filler bytes**: deterministic committed bytes used only for minimal end-of-region alignment when an exact fit is impossible.
- **Fill ratio**: `covered_bytes / usable_region_bytes`.
- **Real-PoRep ratio**: `real_porep_bytes / covered_bytes`.
- **Lease**: a verifier-owned region handle granted to the prover.
- **Rechallenge**: an outer challenge against a previously materialized in-memory object without rebuilding it.

---

## 8. High-level architecture

### 8.1 Components

The repository shall contain these top-level implementation components:

1. **Vendored Filecoin workspace**
   - exact upstream snapshot of the relevant Rust workspace.

2. **Rust bridge**
   - thin wrapper that exposes the official Filecoin sealing and verification path to Python.

3. **Python prover**
   - region intake,
   - session execution,
   - materialization into host/HBM,
   - outer proof generation,
   - timing collection.

4. **Python verifier**
   - region leasing,
   - session planning,
   - inner verification,
   - outer challenge generation and checking,
   - final verdict construction.

5. **CLI**
   - user-facing entrypoint for prover, verifier, and benchmark workflows.

6. **Benchmark harness**
   - named profiles,
   - repeatable measurement runs,
   - artifact production.

7. **Python Filecoin mirror/port**
   - staged deterministic subcomponents and, later, a full Python port gated by parity.

### 8.2 Architectural principle: two proofs, two boundaries

The architecture must preserve two clean boundaries:

- **inner proof boundary**
  - Filecoin sealing and verification logic;
- **outer proof boundary**
  - PoSE-specific region commitment and challenge protocol.

The outer proof must never replace the inner proof, and the inner proof must never be mistaken for proof that the challenged region currently contains the object.

---

## 9. Repository layout

The repository shall use the following structure or a structure equivalent in clarity and separation:

```text
repo/
  README.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  pyproject.toml
  Cargo.toml
  Makefile

  docs/
    repository-spec.md
    architecture.md
    threat-model.md
    protocol.md
    benchmarking.md
    upstream-sync.md
    result-schema.md
    hardware/
      single-h100.md
      eight-h100.md

  vendor/
    rust-fil-proofs/
    UPSTREAM.lock

  rust/
    pose_filecoin_bridge/
    pose_test_hooks/

  proto/
    pose/v1/session.proto

  src/pose/
    __init__.py
    version.py

    cli/
      main.py
      prover.py
      verifier.py
      bench.py

    common/
      errors.py
      hashing.py
      timing.py
      units.py
      env.py
      logging.py

    protocol/
      messages.py
      codec.py
      session_ids.py
      result_schema.py

    filecoin/
      reference.py
      mirror/
        replica_id.py
        parents.py
        labels.py
        comms.py
      port/
        README.md
        experimental/

    prover/
      service.py
      session.py
      planner.py
      object_builder.py
      challenge.py
      cleanup.py
      regions.py
      memory/
        host.py
        gpu.py

    verifier/
      service.py
      policy.py
      deadlines.py
      leasing.py
      challenge.py
      result_writer.py

    benchmarks/
      harness.py
      profiles.py
      summarize.py

  bench_profiles/
    dev-small.yaml
    single-h100-host-max.yaml
    single-h100-hbm-max.yaml
    single-h100-hybrid-max.yaml
    eight-h100-hbm-max.yaml
    eight-h100-hybrid-max.yaml

  scripts/
    sync_upstream.sh
    gen_test_vectors.py
    run_lab_matrix.sh

  tests/
    unit/
    parity/
    integration/
    e2e/
    adversarial/
    hardware/
    performance/
```

### 9.1 Separation invariant

The verifier package must not import prover-only modules.  
The prover package must not contain verifier decision logic.  
Shared data structures must live only in `src/pose/protocol/` and `src/pose/common/`.

---

## 10. Upstream Filecoin integration requirements

### 10.1 Vendoring rule

The repository must vendor the **actual upstream Rust workspace snapshot**, not a copy-pasted subset of source files.

### 10.2 Pinned upstream snapshot

The repository must include `vendor/UPSTREAM.lock` containing at least:

- upstream repository URL;
- upstream commit SHA;
- upstream tag if any;
- sync date;
- local patch status.

### 10.3 Patch policy

Direct edits under `vendor/rust-fil-proofs/` are forbidden by default.

If an emergency patch is required:

- it must be applied through an explicit patch mechanism;
- the patch must be documented in `docs/upstream-sync.md`;
- CI must fail if the patch is not explicitly acknowledged;
- parity and upstream tests must still pass.

### 10.4 Thin bridge rule

The Rust bridge must be thin. It may:

- construct configs,
- call official sealing functions,
- call official verification functions,
- expose deterministic checkpoints for parity tests,
- normalize errors,
- return timing data.

It must not:

- replace the core proving algorithm with local approximations;
- silently fall back to fake/synthetic shortcuts;
- mutate upstream semantics.

### 10.5 Production shortcut ban

Production code must not call any shortcut or fake-proof path, including but not limited to:

- `fauxrep`
- `fauxrep2`
- `fauxrep_aux`
- synthetic proof helpers
- any “fake seal” or “fake proof” benchmark-only path

CI shall include a forbidden-symbol scan that fails if production code references banned paths.

---

## 11. Python Filecoin implementation policy

### 11.1 What must be Python from day one

Python must own:

- prover/verifier process lifecycle;
- CLI;
- region planning and allocation orchestration;
- outer proof implementation;
- result reporting;
- benchmark harness;
- deterministic Filecoin mirrors needed for parity.

### 11.2 Staged port policy

The repository must include a staged Python port under `src/pose/filecoin/port/`.

Promotion from Rust-backed to Python-implemented components shall happen only when the specific component:

1. has a clear reference boundary;
2. has exhaustive parity vectors;
3. passes deterministic equivalence tests;
4. does not weaken the security claim.

### 11.3 Initial Python mirror scope

The initial Python mirror should include deterministic pieces only:

- replica-id derivation;
- DRG parent generation;
- expander parent generation;
- label derivation where feasible;
- commitment assembly logic;
- canonical serialization logic for PoRep units and region manifests.

### 11.4 Full Python port aspiration

A full Python PoRep port may be pursued, but the repository’s production claim of “real Filecoin PoRep” shall remain tied to the vendored upstream path until the Python port has a formal promotion decision backed by parity results.

---

## 12. Canonical PoRep unit

### 12.1 Definition

A **PoRep unit** is the atomic real Filecoin object stored in memory. A PoRep unit must correspond to one real upstream sealing run.

### 12.2 Required fields

Each PoRep unit must include, at minimum:

- protocol version;
- upstream snapshot identifier;
- proof type / PoRep config identifier;
- sector size;
- prover id;
- sector id;
- ticket;
- seed;
- piece information;
- `comm_d`;
- `comm_r`;
- seal proof bytes;
- manifest of included auxiliary artifacts;
- timing information for the inner proof phases.

### 12.3 Storage profiles

The repository shall support these PoRep unit storage profiles:

#### `minimal`
Contains:
- manifest,
- public inputs,
- proof bytes,
- verification-relevant metadata.

Use only for CI and debugging.  
Not suitable for maximum-coverage PoSE.

#### `replica`
Contains:
- everything in `minimal`;
- sealed replica bytes.

Suitable for realistic memory coverage.

#### `full-cache` (default production profile)
Contains:
- everything in `replica`;
- proving auxiliaries and cache artifacts required to avoid recomputation where available.

This is the default because it maximizes real-PoRep occupancy.

### 12.4 Canonical serialization

PoRep units must have a deterministic serialization.

The required format is:

1. **manifest** encoded in deterministic CBOR;
2. **blob table** with deterministic ordering;
3. **blob payloads** concatenated in manifest order;
4. **alignment** to the configured region leaf size;
5. **per-blob SHA-256 digests** recorded in the manifest.

The ordering of blob kinds must be fixed and versioned.

### 12.5 Allowed blob kinds

The manifest may include these blob kinds:

- `seal_proof`
- `sealed_replica`
- `tree_c`
- `tree_r_last`
- `persistent_aux`
- `temporary_aux`
- `labels`
- `cache_file`
- `public_inputs`
- `proof_metadata`

Blob kinds must be explicitly labeled. Unknown blob kinds must cause parse failure unless the session version explicitly allows them.

---

## 13. Region payload

### 13.1 Definition

A **region payload** is the exact sequence of bytes written into one challenged region.

### 13.2 Composition

A region payload shall be:

- one or more serialized PoRep units;
- followed, only if necessary, by a minimal deterministic tail filler.

### 13.3 Tail filler rule

Tail filler is allowed only to satisfy final alignment or unavoidable packing slack.

Constraints:

- tail filler must be deterministic and session-bound;
- tail filler must be derived from:
  - session nonce,
  - region identifier,
  - session plan root;
- tail filler must be clearly counted separately from real-PoRep bytes;
- tail filler must be limited to the smaller of:
  - one outer Merkle leaf,
  - 1 MiB.

The default goal is **zero tail filler**.

### 13.4 Real-PoRep ratio requirement

The repository shall report:

- `real_porep_bytes`
- `tail_filler_bytes`
- `real_porep_ratio`

A session may not report “success” unless `real_porep_ratio` meets the profile threshold.  
Default threshold: **0.99**.

---

## 14. Outer proof system

### 14.1 Purpose

The outer proof proves timely possession of the bytes committed in the challenged region payload.

### 14.2 Commitment

Each region shall have its own Merkle root over the raw region payload bytes.

The session shall also have a session manifest root covering:

- session parameters;
- region manifests;
- region roots;
- region sizes;
- payload profile;
- deadline policy;
- nonce.

### 14.3 Leaf size

The outer proof leaf size must be configurable.

Recommended defaults:

- correctness / CI: `4 KiB`
- large-memory benchmarking: `1 MiB`

The leaf size used in a session must be part of the session manifest.

### 14.4 Region challenge sampling

The verifier shall sample leaves independently per region.

The challenge count must be configurable by a statistical policy:

- target minimum missing fraction `epsilon`;
- target soundness `lambda` bits.

The default challenge policy should compute:

`k = ceil( ln(2^-lambda) / ln(1 - epsilon) )`

and then cap, batch, or range-aggregate challenges only in ways that preserve the documented soundness target.

### 14.5 Batched proofs

To keep challenge traffic practical, the protocol may batch leaves and branches, but batching must not change the semantics of the proof:

- the verifier must still validate membership for each challenged leaf;
- the timing window must include the batched response.

### 14.6 Deadline enforcement

The verifier must reject if:

- the prover misses the outer challenge deadline;
- any challenged opening is invalid;
- the openings are inconsistent with the session manifest.

### 14.7 Rechallenge support

A rechallenge is valid only if:

- the same session manifest root is reused;
- the verifier clearly labels the run as `rechallenge`;
- the result artifact distinguishes rechallenge from full rebuild runs.

---

## 15. Memory model

## 15.1 Common requirements

The repository shall operate only on **user-space accessible memory**.

All memory accounting must distinguish:

- total detected capacity,
- reserved bytes,
- usable bytes,
- covered bytes.

### 15.2 Host memory backend

The host backend shall support:

- anonymous shared mappings or `memfd`;
- optional `mlock`;
- optional huge pages;
- NUMA affinity controls where practical;
- explicit zeroization on teardown.

The verifier creates the host region and pass a lease handle to the prover.

### 15.3 GPU HBM backend

The HBM backend shall support:

- device memory allocations only;
- no unified/managed memory in production mode;
- CUDA IPC handle transfer;
- explicit zeroization on teardown where supported;
- per-device accounting.

The prover must write the PoRep region payload into the leased HBM allocation itself. A host-only copy does not satisfy HBM coverage.

### 15.4 Region planner

The planner shall maximize fill subject to reserve policy.

Inputs:

- available host bytes;
- per-GPU available HBM bytes;
- guard/reserve bytes;
- PoRep unit profile;
- target fill ratio;
- supported sector sizes;
- measured artifact footprints.

Outputs:

- region plan;
- PoRep unit packing plan;
- expected real-PoRep ratio;
- expected slack.

### 15.5 Artifact footprint accounting

The planner must use **measured** unit footprints, not only nominal sector size.

A unit’s in-memory footprint may include:

- sealed replica;
- proof bytes;
- metadata;
- included cache artifacts.

---

## 16. Prover/verifier protocol

### 16.1 Transport

The default transport shall be gRPC over:

- Unix domain sockets for same-host runs; or
- loopback TCP if Unix sockets are unavailable.

The protocol must be versioned.

### 16.2 Message flow

The normative message flow is:

1. `Discover`
2. `PlanSession`
3. `LeaseRegions`
4. `GenerateInnerPoRep`
5. `MaterializeRegionPayloads`
6. `CommitRegions`
7. `VerifyInnerProofs`
8. `ChallengeOuter`
9. `VerifyOuter`
10. `Finalize`
11. `Cleanup`

### 16.3 PlanSession

The verifier creates a session plan containing:

- session id;
- nonce;
- region plan;
- PoRep unit profile;
- sector plan;
- leaf size;
- challenge policy;
- deadlines;
- cleanup policy.

### 16.4 LeaseRegions

The verifier sends lease handles for each region.

Each lease must include:

- region id;
- region type (`host` or `gpu`);
- usable bytes;
- lease handle;
- lease expiry;
- cleanup policy.

### 16.5 GenerateInnerPoRep

The prover generates real Filecoin PoRep units using the vendored bridge.

The prover must record per-unit and per-phase timings.

### 16.6 MaterializeRegionPayloads

The prover:

- serializes PoRep units canonically;
- packs them into the target region;
- writes the region payload into the leased allocation;
- computes the region Merkle root over those bytes;
- returns region manifests and region roots.

### 16.7 VerifyInnerProofs

The verifier verifies all required inner proofs before accepting the region payload commitments.

If inner verification fails, the session fails immediately.

### 16.8 ChallengeOuter

The verifier sends random per-region leaf challenges.

The prover responds with:

- challenged leaf bytes;
- Merkle authentication paths;
- response timing metadata.

### 16.9 Finalize

The verifier computes the final verdict and writes the canonical result artifact.

### 16.10 Cleanup

The cleanup phase shall:

- zero challenged regions according to policy;
- drop leases;
- release handles;
- write cleanup status into the result artifact.

Cleanup failure must be reported even if the proof succeeded.

---

## 17. Session result and verdicts

### 17.1 Allowed verdicts

A session verdict must be one of:

- `SUCCESS`
- `INNER_PROOF_INVALID`
- `OUTER_PROOF_INVALID`
- `TIMEOUT`
- `COVERAGE_BELOW_THRESHOLD`
- `RESOURCE_FAILURE`
- `CLEANUP_FAILURE`
- `PROTOCOL_ERROR`

### 17.2 Required output fields

Every verifier run must emit, at minimum:

- `success`
- `verdict`
- `session_id`
- `profile_name`
- `host_total_bytes`
- `host_usable_bytes`
- `host_covered_bytes`
- `gpu_devices`
- `gpu_usable_bytes_by_device`
- `gpu_covered_bytes_by_device`
- `real_porep_bytes`
- `tail_filler_bytes`
- `real_porep_ratio`
- `coverage_fraction`
- `inner_filecoin_verified`
- `outer_pose_verified`
- `challenge_leaf_size`
- `challenge_count`
- `deadline_ms`
- `response_ms`
- `cleanup_status`
- `timings_ms`
- `environment`

### 17.3 Timing breakdown

`timings_ms` must include:

- `discover`
- `region_leasing`
- `allocation`
- `data_generation`
- `seal_pre_commit_phase1`
- `seal_pre_commit_phase2`
- `seal_commit_phase1`
- `seal_commit_phase2`
- `object_serialization`
- `copy_to_host`
- `copy_to_hbm`
- `outer_tree_build`
- `inner_verify`
- `challenge_response`
- `outer_verify`
- `cleanup`
- `total`

### 17.4 Result artifact example

```json
{
  "success": true,
  "verdict": "SUCCESS",
  "session_id": "2026-03-20T12:00:00Z-8e2d7d2a",
  "profile_name": "single-h100-hybrid-max",
  "host_total_bytes": 1030792151040,
  "host_usable_bytes": 515396075520,
  "host_covered_bytes": 510027366400,
  "gpu_devices": [0],
  "gpu_usable_bytes_by_device": {"0": 79691776000},
  "gpu_covered_bytes_by_device": {"0": 78926315520},
  "real_porep_bytes": 588953681920,
  "tail_filler_bytes": 1048576,
  "real_porep_ratio": 0.9999982,
  "coverage_fraction": 0.9892,
  "inner_filecoin_verified": true,
  "outer_pose_verified": true,
  "challenge_leaf_size": 1048576,
  "challenge_count": 768,
  "deadline_ms": 2200,
  "response_ms": 731,
  "cleanup_status": "ZEROIZED_AND_RELEASED",
  "timings_ms": {
    "discover": 11,
    "region_leasing": 34,
    "allocation": 118,
    "data_generation": 43,
    "seal_pre_commit_phase1": 12984,
    "seal_pre_commit_phase2": 6321,
    "seal_commit_phase1": 14021,
    "seal_commit_phase2": 9211,
    "object_serialization": 1880,
    "copy_to_host": 940,
    "copy_to_hbm": 1710,
    "outer_tree_build": 6033,
    "inner_verify": 818,
    "challenge_response": 731,
    "outer_verify": 202,
    "cleanup": 141,
    "total": 53458
  },
  "environment": {
    "python_version": "3.x",
    "rust_toolchain": "pinned",
    "cuda_runtime": "captured",
    "driver_version": "captured",
    "upstream_commit": "captured"
  }
}
```

---

## 18. CLI requirements

### 18.1 Commands

The CLI shall expose three top-level personas.

#### Prover

```bash
pose prover serve --config prover.toml
pose prover inspect
pose prover self-test
```

#### Verifier

```bash
pose verifier run --profile single-h100-hybrid-max
pose verifier run --plan plan.yaml --json
pose verifier rechallenge --session-id <id>
pose verifier verify-record result.json
```

#### Benchmark

```bash
pose bench run --profile single-h100-hbm-max
pose bench matrix --profiles bench_profiles/
pose bench summarize results/*.json
```

### 18.2 Exit codes

The CLI must use stable exit codes:

- `0` success;
- nonzero for all failures.

The human-readable summary must never be the only output path; JSON must always be available.

### 18.3 Human-readable summary

The human summary must include:

- session verdict;
- host and GPU covered bytes;
- fill ratio;
- real-PoRep ratio;
- total runtime;
- challenge response time;
- inner verify status;
- outer verify status.

---

## 19. Test suite

The test suite is mandatory and central to repository credibility.

### 19.1 Test categories

The repository shall contain:

- `unit`
- `parity`
- `integration`
- `e2e`
- `adversarial`
- `hardware`
- `performance`

### 19.2 Upstream integrity tests

These tests prove that the repository still uses the real upstream implementation.

Required checks:

- `vendor/UPSTREAM.lock` matches the vendored tree;
- production code does not reference banned shortcut APIs;
- the vendored Rust workspace passes its own test suite in CI;
- upstream patch status is clean or explicitly acknowledged.

### 19.3 Parity tests: Python mirror vs Rust reference

Parity tests must compare Python outputs against the vendored Rust reference for deterministic components, including:

- replica-id derivation;
- DRG parent generation;
- expander parent generation;
- label-related checkpoints where exposed;
- `comm_d`;
- `comm_r`;
- `tree_c` root where exposed;
- `tree_r_last` root where exposed;
- manifest serialization;
- region payload root.

The expected result is either:

- bit-for-bit equality; or
- explicitly documented semantic equivalence for proof artifacts where bytes may differ but verification semantics are identical.

### 19.4 Inner proof correctness tests

Required cases:

- small CI sectors;
- multiple supported sector shapes in nightly;
- lab-only large-sector runs on H100 hardware.

Tests must cover:

- successful seal + verify;
- tampered `comm_d` failure;
- tampered `comm_r` failure;
- wrong prover id failure;
- wrong sector id failure;
- wrong randomness failure;
- wrong proof bytes failure.

### 19.5 Outer proof correctness tests

Required cases:

- valid region root and openings;
- tamper one byte -> reject;
- wrong leaf index -> reject;
- wrong branch -> reject;
- wrong region id -> reject;
- stale session nonce -> reject;
- stale manifest root -> reject.

### 19.6 Protocol tests

Required cases:

- version mismatch handling;
- malformed message handling;
- timeout handling;
- duplicate challenge handling;
- repeated session id rejection;
- verifier/prover restart recovery where applicable.

### 19.7 Separation tests

Required invariants:

- verifier package has no imports from prover-only modules;
- prover package has no verifier-policy logic;
- bridge interface is narrow and stable;
- result schema round-trips cleanly.

### 19.8 Adversarial tests

The adversarial suite must explicitly test:

- old valid inner proof replayed under a new session nonce;
- correct inner proof with incorrect outer bytes;
- object stored only on disk or non-challenged memory;
- partial overwrite of a leased region;
- HBM region not actually filled;
- host region with sparse writes;
- timeout caused by post-challenge copy-in;
- insufficient coverage despite valid proof bytes elsewhere;
- mismatch between declared and actual payload length.

### 19.9 Hardware tests

Self-hosted or lab hardware jobs must include:

- single-H100 host-only;
- single-H100 HBM-only;
- single-H100 hybrid;
- 8×H100 HBM-only;
- 8×H100 hybrid.

Every hardware job must archive:

- result JSON;
- benchmark logs;
- environment snapshot;
- upstream commit;
- Rust toolchain version;
- CUDA runtime/driver versions;
- GPU inventory.

### 19.10 Performance regression tests

Nightly performance tests must detect regressions in:

- inner proof timings;
- region payload construction;
- HBM transfer time;
- outer tree build time;
- challenge response time;
- overall coverage fraction.

---

## 20. Benchmarking

### 20.1 Benchmark classes

The benchmark harness shall support:

- `cold`
- `warm`
- `rechallenge`

Definitions:

- **cold**: includes first-run setup costs;
- **warm**: repeated steady-state runs;
- **rechallenge**: no rematerialization, only re-openings on resident payloads.

### 20.2 Required benchmark profiles

The repository must include these named profiles:

- `dev-small`
- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`
- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

### 20.3 Benchmark profile fields

Each profile must define, at minimum:

- target devices;
- reserve policy;
- host target fraction;
- per-GPU target fraction;
- PoRep unit profile;
- leaf size;
- challenge policy;
- deadline policy;
- cleanup policy;
- repetition count.

### 20.4 Reported benchmark metrics

Every benchmark summary must report:

- success rate;
- mean / p50 / p95 / p99 timings;
- deadline miss rate;
- coverage fraction;
- real-PoRep ratio;
- per-device HBM coverage;
- verifier CPU time;
- rechallenge performance.

---

## 21. Build and packaging

### 21.1 Python packaging

Use a standard Python package layout with pinned dependencies.

### 21.2 Rust packaging

Use a workspace or independent crate arrangement that makes the bridge reproducible.

### 21.3 Python/Rust integration

The preferred integration mechanism is `pyo3` + `maturin`.

### 21.4 Reproducibility

The repository must have:

- pinned Python dependencies;
- a Cargo lockfile;
- reproducible local build instructions;
- a single-command test path;
- a single-command benchmark path.

### 21.5 Required developer commands

At minimum:

```bash
make sync-upstream
make build
make test
make test-parity
make test-hardware
make bench PROFILE=single-h100-hbm-max
```

---

## 22. Documentation requirements

The repository must ship the following documents:

- `docs/repository-spec.md`
- `docs/architecture.md`
- `docs/threat-model.md`
- `docs/protocol.md`
- `docs/benchmarking.md`
- `docs/upstream-sync.md`
- `docs/result-schema.md`

Each document must be consistent with this spec.  
If a document disagrees with this file, this file wins.

---

## 23. Milestones

### 23.1 Phase 0 — foundation

Deliverables:

- vendored upstream workspace;
- Rust bridge skeleton;
- basic Python package structure;
- upstream integrity CI;
- parity harness scaffolding.

Exit criteria:

- upstream tests run in CI;
- bridge can execute one real seal + verify flow.

### 23.2 Phase 1 — host-memory PoSE

Deliverables:

- local-leased host memory regions;
- canonical PoRep unit serialization;
- outer host-region proof;
- verifier result artifact;
- host-only CLI flow.

Exit criteria:

- successful host-only PoSE on development hardware;
- adversarial host-memory tests passing.

### 23.3 Phase 2 — single-H100 HBM PoSE

Deliverables:

- HBM leasing via CUDA IPC;
- HBM materialization path;
- HBM outer proof;
- single-H100 benchmark profiles.

Exit criteria:

- successful HBM-only PoSE on one H100;
- HBM challenge response benchmarks archived.

### 23.4 Phase 3 — hybrid host + HBM

Deliverables:

- mixed-region sessions;
- unified result reporting;
- planner across host + HBM;
- hybrid benchmark profiles.

Exit criteria:

- successful hybrid session on one H100 box.

### 23.5 Phase 4 — 8×H100 scale-out

Deliverables:

- per-device worker model;
- region aggregation across 8 GPUs;
- multi-GPU benchmark harness;
- lab automation.

Exit criteria:

- successful 8×H100 HBM-only and hybrid sessions.

### 23.6 Phase 5 — Python port promotion

Deliverables:

- deterministic Python mirror parity closure;
- formal promotion process for selected components.

Exit criteria:

- promoted Python components have parity reports and no regression in security claims.

---

## 24. Acceptance criteria

The repository shall not be considered complete until all of the following are true:

1. A verifier can run a successful end-to-end PoSE against host memory.
2. A verifier can run a successful end-to-end PoSE against HBM on a single H100.
3. The same protocol scales to an 8×H100 box.
4. Every successful session includes at least one real Filecoin proof accepted by the official verification path.
5. Production mode contains no shortcut or fake Filecoin proof path.
6. Deterministic Python mirrors pass parity tests against vendored Rust.
7. CLI output includes success/failure and per-phase timings.
8. Benchmark profiles exist and produce machine-readable artifacts.
9. The default benchmark profiles maximize challenged user-space memory subject to reserve policy.
10. The repository reports coverage and real-PoRep ratio honestly and separately.

---

## 25. Design invariants

The following invariants must remain true unless this spec is revised:

1. **Real Filecoin first.**  
   The inner proof path is the actual upstream Filecoin implementation until formally replaced.

2. **Two-proof model.**  
   PoSE is the conjunction of inner Filecoin verification and outer timed storage verification.

3. **Verifier/prover separation.**  
   The verifier remains a distinct process with distinct responsibilities.

4. **No silent downgrades.**  
   Production mode must never silently fall back to fake proofs, synthetic proofs, managed memory, or weaker protocol modes.

5. **Coverage honesty.**  
   All reported memory coverage must reflect actual challenged user-space bytes, not nominal hardware capacity.

6. **Parity before promotion.**  
   No Python Filecoin component may replace the reference path without passing parity gates.

---

## 26. Open implementation notes

This section is informative.

- The initial repository should optimize for clarity and evidence, not micro-optimizations.
- The strongest first implementation target is `local-leased` same-host benchmarking on one H100.
- The fastest path to credibility is:
  1. vendor upstream,
  2. bridge official seal/verify,
  3. build the outer leased-region protocol,
  4. prove host-only first,
  5. then HBM,
  6. then 8×H100.
- The benchmark harness should preserve enough metadata that performance history is scientifically useful rather than anecdotal.

---

## 27. Summary

The repository shall be a Python-first PoSE system with:

- a real vendored Filecoin PoRep core,
- a clean prover/verifier process split,
- leased-region host and HBM challenges,
- a canonical serialized PoRep object,
- a timed outer proof of storage,
- strong benchmark and parity infrastructure,
- and an explicit path from single-H100 to 8×H100 deployment.

That is the required shape of the implementation.
