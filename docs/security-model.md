# Security Model

## Scope

This repository targets the graph-based PoSE-DB protocol described in the
bundled paper and operationalized by `repository-spec.md`.

The repository reports two different claim classes and must not blur them:

- formal claim:
  theorem-based storage claim under the paper's model;
- operational claim:
  engineering claim about which verifier-leased host/HBM regions were used as
  the canonical storage location.

## Assumptions

The security story depends on all of the following:

- verifier-owned challenged regions;
- timed fast rounds with conservative deadline calibration;
- the distant-attacker interpretation used by the paper;
- the random-oracle heuristic for the selected concrete hash backend;
- honest reporting of attacker-accessible local memory budget `M`.

## Attacker Budget

All local memory that could help the prover during the fast phase must be
counted toward `M` unless separately constrained and documented.

This includes, when applicable:

- unchallenged DRAM;
- unchallenged HBM;
- pinned host buffers;
- stage buffers;
- managed-memory mirrors;
- fast local scratch files or mappings.

Tier-specific marketing claims are out of scope unless the repository explicitly
adds stronger mechanisms and documents them separately.

## Timed Fast Phase

The fast phase is the core security-critical path.

For each round:

- the verifier samples one challenge index uniformly from `[0, m)`;
- the prover returns the label stored in that slot;
- the verifier checks correctness and the per-round deadline `Delta`.

Production profiles are valid only when calibration establishes a conservative
`q` such that `q < gamma`.

## Trusted Computing Base

The trusted computing base includes:

- the verifier process;
- the operating system kernel;
- the region-leasing primitives;
- the Python reference graph and label semantics;
- any enabled native acceleration modules;
- the selected hash backend;
- CUDA runtime and driver behavior in HBM mode;
- the verifier's timing source and instrumentation.

## Non-Goals

This repository does not claim:

- kernel-level or firmware-level erasure;
- secure hardware attestation;
- protection against a malicious kernel, driver, or device firmware;
- storage claims for inaccessible or privileged memory;
- disk or remote-storage erasure.
