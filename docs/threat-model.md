# Threat Model

## Security Claim

A session succeeds only when the prover both:

- demonstrates a real Filecoin PoRep accepted by the official verification
  path; and
- demonstrates timely possession of the canonical serialized PoRep bytes inside
  the verifier-leased host or HBM regions.

## Attacker Model

The prover may be adversarial and may attempt to:

- replay an old but valid inner proof under a new session;
- store a different object than the claimed PoRep bytes;
- keep the object outside the challenged region and answer leaves on demand;
- partially overwrite the region or sparsely populate it;
- exploit transport ambiguity, stale manifests, or stale lease handles;
- answer after an excessive delay that hides reconstruction or copy costs.

## Trusted Components

The trusted computing base includes:

- the verifier process;
- the host kernel and the region-leasing primitives it uses;
- the vendored Filecoin reference implementation;
- the Rust bridge surface;
- the Python verifier implementation;
- CUDA runtime and NVIDIA driver behavior in HBM mode;
- the cryptographic hashing used for the outer proof.

## Explicit Non-Goals

The repository does not claim:

- kernel, firmware, or device-wide erasure;
- erasure of inaccessible memory;
- protection against a malicious kernel or driver stack;
- disk or remote-storage erasure;
- blockchain integration.

## Timing Assumption

Deadline policies rely on an operational assumption:

- reading and proving bytes already resident in the challenged region must be
  materially cheaper than reconstructing them or copying them in after the
  challenge arrives.

Benchmarking must therefore preserve separate measurements for:

- resident challenge response;
- copy-in from alternate storage;
- reconstruction or rematerialization.

