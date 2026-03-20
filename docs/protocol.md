# Protocol

## Transport

The target transport is versioned gRPC over Unix domain sockets by default, with
loopback TCP allowed when Unix sockets are unavailable.

The current repository foundation ships the protocol schema and Python message
types, but not a production transport implementation yet.

## Normative Session Flow

The planned session flow follows the specification exactly:

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

Every verifier run must produce:

- a machine-readable JSON result artifact;
- a stable verdict;
- coverage and real-PoRep accounting;
- per-phase timings;
- environment metadata.

The current foundation provides a Python result schema module that encodes these
required fields so later phases can build on a stable contract.

