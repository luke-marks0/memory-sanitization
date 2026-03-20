# Milestones

## Current State

- Foundation layout: complete
- Official upstream vendoring and integrity controls: complete
- Vendored upstream Rust workspace validation: complete
- Real Python-callable vendored Filecoin bridge: complete
- Canonical PoRep unit serialization: complete on the Python-owned path
- Host-memory PoSE: complete
- HBM and hybrid modes: not started
- Scale-out and parity-gated promotion: not started

## Milestone 1: Phase 0 Completion

Goal:
Deliver a real Python-to-Rust bridge for the vendored Filecoin implementation
and prove one real seal plus verify flow end to end.

Tasks:
- expose a thin `pyo3` bridge over the vendored upstream seal and verify path;
- provide a Python adapter that can call the extension without placeholder
  semantics on the happy path;
- add repository commands to build the extension and run a real bridge smoke
  test;
- add CI coverage for one real upstream-backed seal plus verify run.

Acceptance:
- `make build` compiles the bridge;
- `make test-bridge` passes;
- CI runs a bridge smoke test that seals and verifies one real 2 KiB sector.

Status:
- complete;
- the bridge is built through `scripts/build_bridge.py`;
- `VendoredFilecoinReference` seals and verifies a real upstream-backed 2 KiB
  sector artifact;
- `make test-bridge` and the CI `bridge-smoke` job validate the end-to-end
  bridge path.

## Milestone 2: Canonical PoRep Unit

Goal:
Define the exact bytes that are stored in challenged memory.

Tasks:
- implement the canonical PoRep-unit manifest and serialization format;
- define deterministic blob ordering and tail filler behavior;
- add fixtures and round-trip tests for the serialization contract.

Acceptance:
- the canonical PoRep payload format is stable, documented, and covered by
  tests;
- real vendored bridge artifacts round-trip through the canonical serializer.

Status:
- complete on the Python-owned path;
- the canonical PoRep-unit manifest, deterministic CBOR encoding, fixed blob
  ordering, and aligned byte layout are implemented;
- `minimal`, `replica`, and `full-cache` profiles are modeled, with automatic
  `minimal` unit construction from the real bridge artifact;
- fixtures, tamper checks, and real-bridge round-trip coverage are in place.

## Milestone 3: Host-Memory PoSE

Goal:
Deliver Phase 1 host-only secure-erasure sessions.

Tasks:
- implement verifier-owned host leases;
- materialize canonical PoRep units into leased host memory;
- implement outer commitments, challenges, openings, deadlines, and cleanup;
- produce honest result artifacts and adversarial test coverage.

Acceptance:
- a host-only session succeeds only when both the inner Filecoin proof and the
  outer timed storage proof succeed;
- replay, partial overwrite, wrong-byte, and timeout tests fail correctly.

Status:
- complete;
- `pose verifier run --profile dev-small` executes a real local host-only
  session with a verifier-owned host lease, canonical PoRep-unit materialization,
  deterministic session-plan-bound tail filler, per-region Merkle commitment,
  challenge verification, deadline enforcement, and cleanup;
- the real CLI path now uses the versioned gRPC Unix-socket transport defined in
  `proto/pose/v1/session.proto`, with `pose prover serve --config ...` as the
  prover endpoint and an auto-started ephemeral prover for local runs;
- host sessions now validate the returned region manifest against the actual
  leased bytes before accepting the outer proof;
- `pose verifier run --plan ...` and retained-session rechallenge flows are
  implemented on the production path;
- adversarial host-memory coverage now includes replayed openings under a new
  session nonce, wrong outer bytes, partial overwrite, sparse writes, timeout,
  mismatch between declared and actual payload length, insufficient real-PoRep
  ratio, and cleanup failure reporting;
- Phase 1 is complete and spec-compliant on the current host-only `minimal`
  profile path.

## Milestone 4: Single-H100 HBM

Goal:
Extend the host-only design to verifier-owned CUDA HBM allocations.

Tasks:
- implement CUDA IPC lease handling and HBM accounting;
- materialize payloads into HBM and verify cleanup;
- add benchmark and result reporting for HBM-only runs.

Acceptance:
- one real single-H100 HBM session passes with the same success semantics as
  host mode.

## Milestone 5: Hybrid Host + HBM

Goal:
Support mixed placement across host memory and device memory.

Tasks:
- implement a region planner that uses measured usable memory budgets;
- commit and challenge the full mixed layout;
- report per-region and aggregate coverage honestly.

Acceptance:
- hybrid sessions can be benchmarked and compared directly against host-only
  and HBM-only runs.

## Milestone 6: Scale-Out and Benchmark Compliance

Goal:
Reach the named single-box and 8xH100 benchmark targets from the spec.

Tasks:
- add per-device worker coordination and result aggregation;
- ship the required benchmark profiles as runnable profiles, not placeholders;
- archive benchmark artifacts and summaries.

Acceptance:
- benchmark profiles from the README and spec produce comparable machine-readable
  artifacts with real timing and coverage data.

## Milestone 7: Parity-Gated Python Promotion

Goal:
Expand Python Filecoin mirrors without replacing the real reference path
prematurely.

Tasks:
- generate vectors from the Rust reference implementation;
- expand deterministic Python mirrors under parity tests;
- document explicit promotion gates for any future Python replacement.

Acceptance:
- no production component is promoted out of the Rust reference path without
  parity evidence and an explicit promotion decision.
