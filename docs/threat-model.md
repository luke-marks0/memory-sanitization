# Threat Model

## Security Claim

A successful session means the prover answered timed PoSE-DB challenge rounds
for labels derived from the verifier's session seed and the active graph
descriptor, under the declared attacker-budget assumption.

Operationally, the repository may also report which verifier-leased host/HBM
regions held the canonical slot layout for those labels.

## Attacker Model

The prover may be adversarial and may coordinate with an external conspirator.
It may attempt to:

- store only part of the label set and recompute missing labels on demand;
- keep labels outside the challenged regions and answer from another local tier;
- use hidden host, HBM, pinned, or managed-memory shadows;
- sparsely populate the challenged regions;
- replay stale session seeds or stale label state;
- exploit message ambiguity, stale handles, or transport jitter;
- answer late in a way that hides recomputation or copy-in cost.

## Distant-Attacker Interpretation

The repository follows the paper's distant-attacker interpretation:

- assistance may happen before and between fast rounds;
- only local prover state is relevant during a timed round;
- deadlines and calibrated `q` must make within-round recomputation implausible.

## Trusted Components

The trusted computing base includes:

- the verifier process;
- the operating system kernel;
- memory-sharing and lease primitives;
- the Python reference graph and label semantics;
- any enabled native accelerators;
- the selected hash backend;
- CUDA runtime and NVIDIA driver behavior in HBM mode;
- verifier timing instrumentation.

## Explicit Non-Goals

The repository does not claim:

- kernel-level or firmware-level erasure;
- inaccessible or privileged memory coverage;
- disk or remote-storage erasure;
- protection against a malicious kernel, driver, or device firmware;
- secure-hardware attestation.

## Timing Assumption

The fast phase relies on the assumption that reading already resident labels is
materially cheaper than recomputing or copying them into place after challenge.

Calibration and benchmarking must therefore preserve evidence for:

- resident lookup latency;
- local hash throughput;
- transport overhead;
- alternate-store copy timing where relevant.
