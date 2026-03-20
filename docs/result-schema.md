# Result Schema

## Verdicts

Verifier results must use one of the stable verdicts defined by the spec:

- `SUCCESS`
- `INNER_PROOF_INVALID`
- `OUTER_PROOF_INVALID`
- `TIMEOUT`
- `COVERAGE_BELOW_THRESHOLD`
- `RESOURCE_FAILURE`
- `CLEANUP_FAILURE`
- `PROTOCOL_ERROR`

## Required Fields

The canonical result artifact must include at least:

- session identity and profile name;
- run class, session plan root, and session manifest root;
- host total, usable, and covered bytes;
- GPU device list and per-device usable plus covered bytes;
- `real_porep_bytes`, `tail_filler_bytes`, and `real_porep_ratio`;
- `coverage_fraction`;
- inner and outer verification status;
- challenge leaf size, count, deadline, and response time;
- cleanup status;
- artifact path and, for retained sessions, the resident prover endpoint plus
  lease expiry;
- per-phase timings;
- environment metadata.

## Timing Keys

The timing map must carry stable keys for:

- discovery and region leasing;
- allocation and data generation;
- all inner proof phases;
- serialization and copy operations;
- outer tree build and outer verification;
- challenge response;
- cleanup;
- total runtime.

## Current Status

The Python module `src/pose/protocol/result_schema.py` defines the current
Phase 1 result schema and validation helpers. Later phases may extend behavior
without breaking the required field names or verdict set.
