# Architecture

## Intent

This repository is a Python-first orchestration layer around a Rust-backed
Filecoin PoRep reference path. The design follows the repository specification:

- Python owns process lifecycle, protocol, planning, result capture, and the
  outer proof of storage.
- Rust owns the production inner Filecoin seal and verify path until parity
  gates allow promotion of Python components.
- The verifier and prover remain separate processes with a versioned protocol
  boundary.

## System Boundaries

The implementation is divided into two proof boundaries:

1. Inner proof boundary
   The Rust reference bridge is the only production path for real Filecoin
   sealing and verification.
2. Outer proof boundary
   The Python verifier and prover implement region leasing, region commitments,
   challenge sampling, deadline enforcement, and session verdicts.

These boundaries must not collapse into one another. A valid inner proof is not
enough to prove current storage in the challenged region, and a valid outer
proof is not enough to claim the payload is a real Filecoin artifact.

## Repository Map

- `docs/`
  Architecture, threat model, protocol, benchmarking, upstream sync, and result
  schema documentation.
- `vendor/`
  Vendored upstream Filecoin workspace snapshot and sync lock.
- `rust/`
  Rust workspace skeleton for the Python bridge and deterministic test hooks.
- `proto/`
  Versioned protocol definitions for the prover/verifier boundary.
- `src/pose/`
  Python package skeleton for CLI, common utilities, protocol types, prover,
  verifier, benchmark harness, and deterministic Filecoin mirrors.
- `bench_profiles/`
  Named benchmark profile inputs required by the specification.
- `scripts/`
  Repository maintenance and lab automation entrypoints.
- `tests/`
  Mandatory test category scaffolding.

## Implementation Phases

The spec-defined order remains the execution order:

1. Vendor upstream Filecoin Rust and record provenance.
2. Build the thin bridge and prove one real seal plus verify flow.
3. Stand up Python prover and verifier process skeletons.
4. Implement host-memory leased-region PoSE.
5. Extend to single-H100 HBM.
6. Extend to hybrid host plus HBM sessions.
7. Scale to 8xH100.
8. Expand parity coverage and only then consider Python component promotion.

## Current Status

This document reflects the completed Phase 1 host-only plus Phase 2
single-H100 HBM state:

- repository layout now matches the normative shape;
- documentation set exists and is aligned to the spec;
- the official upstream `rust-fil-proofs` workspace is vendored, pinned, and
  reproducibly revalidated;
- upstream Rust workspace tests can be bootstrapped and run from this
  repository;
- the Rust bridge owns the first real Python-callable seal and verify path for a
  2 KiB sector;
- canonical PoRep-unit serialization is now defined and implemented on the
  Python-owned path;
- a local host-only verifier session now exists for the implemented `minimal`
  host profile path, including verifier-owned host leasing, deterministic
  session-plan-bound tail filler, Merkle commitment, challenge openings,
  deadline enforcement, and cleanup;
- the host-only CLI path now uses the versioned gRPC Unix-socket protocol from
  `proto/pose/v1/session.proto`; the verifier keeps policy, challenge
  selection, manifest validation, and verdict logic while the prover service
  owns inner-proof generation, materialization, and challenge openings;
- the same gRPC boundary now supports verifier-owned CUDA IPC HBM leases for a
  single gpu region, including prover-side materialization into HBM,
  verifier-side payload validation, outer HBM openings, and zeroized cleanup;
- the benchmark harness now archives host-only and single-gpu HBM benchmark
  bundles with result JSON, summary metrics, logs, environment metadata,
  upstream/toolchain information, and GPU inventory;
- the host session result artifact now records both `session_plan_root` and
  `session_manifest_root`, and retained sessions record the resident prover
  endpoint needed for rechallenge;
- adversarial host-memory tests cover replayed openings, wrong outer bytes,
  partial overwrite, sparse writes, timeout, payload-length mismatch,
  insufficient real-PoRep ratio, and cleanup failure reporting;
- adversarial HBM coverage now includes rejection of sessions where the leased
  device allocation is not actually filled with the declared payload;
- `pose verifier run --plan ...` and `pose verifier rechallenge --session-id ...`
  are implemented on the real host-only path;
- `pose verifier run --profile single-h100-hbm-max` and gpu-only
  `pose verifier run --plan ...` are implemented on the real single-gpu HBM
  path;
- Phase 0 is complete, Phase 1 is complete on the current host-only `minimal`
  profile path, and the Phase 2 single-gpu HBM code path plus benchmark
  archive machinery are implemented pending a real archived single-H100 lab
  run;
- later hybrid, scale-out, and parity-gated promotion work remains ahead.
