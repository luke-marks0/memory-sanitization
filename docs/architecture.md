# Architecture

## Intent

The target architecture is a Python-first PoSE-DB system over verifier-leased
host memory and GPU HBM.

The authoritative semantics live in Python:

- graph construction;
- graph descriptor encoding;
- challenge-set ordering;
- label derivation;
- soundness calculations;
- verifier decisions and reporting.

Native acceleration is optional and must remain semantically identical to the
Python reference implementation.

## Architectural Rule

The repository implements one protocol with two phases:

1. initialization:
   derive and materialize the session label array from the verifier's seed and
   session parameters;
2. fast phase:
   run timed challenge-response rounds over resident labels.

The design must not split back into separate proof layers.

## Components

- `docs/`
  Normative and implementation-facing documentation.
- `docs/references/`
  Bundled paper and supporting references.
- `proto/`
  Versioned prover/verifier protocol definitions.
- `src/pose/common/`
  Shared utilities and canonical encodings.
- `src/pose/graphs/`
  Reference graph construction and ordering semantics.
- `src/pose/hashing/`
  Random-oracle backend plumbing and input encoding.
- `src/pose/prover/`
  Prover-side session execution and memory materialization.
- `src/pose/verifier/`
  Verifier-side planning, leasing, deadlines, soundness, and verdict logic.
- `src/pose/benchmarks/`
  Profiles, calibration, benchmark execution, and summaries.
- `tests/`
  Conformance, parity, integration, adversarial, hardware, and performance
  coverage.

## Process Split

The verifier and prover remain separate processes.

The verifier owns:

- session planning;
- region leasing;
- challenge generation;
- deadline enforcement;
- expected-label preparation;
- soundness reporting;
- final verdict construction.

The prover owns:

- graph traversal during materialization;
- in-place label generation;
- writes into leased regions;
- timed label lookup during the fast phase;
- cleanup of its temporary state.

## Current Migration State

The repository is currently mid-cutover from an older design.

Supported repository evolution should now follow the PoSE-DB spec only:

- new code should be added under PoSE-DB concepts;
- migration work should delete old protocol assumptions rather than preserve
  them behind compatibility layers;
- the checklist in `migration-checklist.md` is the execution tracker for the
  cutover.
