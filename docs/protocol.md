# Protocol

## Transport

The control plane uses versioned same-host IPC, with gRPC over Unix domain
sockets as the default baseline.

The fast phase is separate in concept even if it temporarily shares the same
transport implementation during migration. Production profiles are valid only if
their calibration includes the real transport overhead and still establishes
`q < gamma`.

## Normative Session Flow

The target PoSE-DB session flow is:

1. `Discover`
2. `PlanSession`
3. `LeaseRegions`
4. `SeedSession`
5. `MaterializeLabels`
6. `PrepareFastPhase`
7. `RunFastPhase`
8. `Finalize`
9. `Cleanup`

## Session Planning

The verifier owns session planning.

The session plan must bind at minimum:

- session ID and session seed;
- graph family and graph parameters;
- `m`, `gamma`, `w_bits`, and hash backend;
- region plan and slot layout;
- adversary model and attacker-budget assumption;
- round count `r`, deadline `Delta`, and calibrated `q`;
- cleanup policy.

## Challenged State

The covered state is the label array:

`sigma = l(o1) || l(o2) || ... || l(om)`

Each challenge index identifies one slot in that array and therefore one
physical slot in one verifier-leased region.

## Lease Ownership

The verifier allocates challenged regions and leases them to the prover.

- host mode uses verifier-owned shared mappings or equivalent handles;
- HBM mode uses verifier-owned CUDA allocations exported through CUDA IPC.

The lease boundary is the basis for operational claim reporting.

## Fast Phase

For each round:

- the verifier samples one challenge index uniformly from `[0, m)`;
- the verifier starts the round timer immediately before challenge emission;
- the prover returns the label bytes stored in that slot;
- the verifier rejects the round if the response is wrong or late.

## Output Requirements

Every verifier run must emit a machine-readable result artifact containing at
least:

- verdict and success bit;
- graph/session parameters;
- attacker-budget assumption and soundness model;
- covered and slack bytes by tier;
- calibrated `q`, `gamma`, and rounds `r`;
- per-phase and fast-phase timing data;
- cleanup status;
- environment metadata.
