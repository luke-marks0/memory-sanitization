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

This document reflects the completed Phase 1 host-only state:

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
- the host-only CLI path uses a prover worker subprocess for materialization and
  challenge responses, while the verifier keeps policy, challenge selection,
  manifest validation, and verdict logic;
- the host session result artifact now records both `session_plan_root` and
  `session_manifest_root`;
- adversarial host-memory tests cover replayed openings, wrong outer bytes,
  partial overwrite, sparse writes, timeout, payload-length mismatch,
  insufficient real-PoRep ratio, and cleanup failure reporting;
- Phase 0 and Phase 1 are complete;
- later HBM, hybrid, scale-out, and parity-gated promotion work remains ahead.
