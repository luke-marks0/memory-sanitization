# Protocol

## Transport

The target transport is versioned gRPC over Unix domain sockets by default, with
loopback TCP allowed when Unix sockets are unavailable.

Phase 1 now implements the same-host production path over versioned gRPC Unix
domain sockets.

- the protobuf schema lives in `proto/pose/v1/session.proto`;
- generated Python bindings live in `src/pose/v1/`;
- `pose prover serve --config prover.toml` runs the prover-side gRPC service;
- `pose verifier run ...` auto-starts an ephemeral prover service for local
  host-only sessions unless the user starts one explicitly.

## Normative Session Flow

The implemented Phase 1 host path follows the spec-defined flow:

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

## Session Planning Requirements

The verifier owns the policy layer and produces a session plan that includes:

- session identity and nonce;
- region plan and lease metadata;
- PoRep unit profile and sector plan;
- challenge policy and deadline policy;
- cleanup policy.

## Lease Ownership Model

The verifier allocates challenged regions and leases them to the prover.

- Host mode uses verifier-owned shared mappings or equivalent handles.
- HBM mode uses verifier-owned CUDA allocations exported through CUDA IPC.

The lease boundary exists so coverage claims remain tied to explicit,
verifier-tracked memory.

## Output Requirements

Every verifier run produces:

- a machine-readable JSON result artifact;
- a stable verdict;
- coverage and real-PoRep accounting;
- per-phase timings;
- environment metadata.

The verifier writes the canonical JSON artifact under `.pose/results/`, prints a
human-readable summary to stderr, and prints the JSON artifact to stdout.
