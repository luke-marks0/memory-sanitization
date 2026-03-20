# pose-filecoin-pose

Python-first proof of secure erasure (PoSE) over host memory and GPU HBM using a **real Filecoin PoRep** as the cryptographic object being stored.

## What this repository is for

This project aims to prove, in a time-bounded way, that a prover process has overwritten challenged **user-space host memory** and/or **GPU HBM** with a **genuine Filecoin proof-of-replication artifact**, and that it can answer verifier challenges against that in-memory object.

The design uses two proofs:

1. **Inner proof:** a real Filecoin PoRep generated and verified through vendored upstream Rust code.
2. **Outer proof:** a timed proof of storage over the bytes actually written into the challenged host/HBM region.

A session succeeds only if **both** proofs succeed.

## Project priorities

In order of importance:

1. **Cryptographic credibility**
   - the PoSE must be a real proof of storage of a real Filecoin PoRep object.

2. **Real Filecoin implementation**
   - the repository vendors the relevant upstream Filecoin Rust workspace and uses it directly in the production inner proof path.

3. **Clean prover/verifier separation**
   - prover and verifier run as separate processes, with a protocol boundary that makes benchmarking and iteration straightforward.

4. **Memory coverage**
   - the system should cover as much user-space accessible host memory and HBM as possible.

5. **Benchmarkability**
   - the CLI and result artifacts must make it easy to compare host-only, HBM-only, and hybrid runs.

## Security model in one page

- the verifier allocates the challenged host memory and/or GPU HBM region;
- the verifier leases that region to the prover process;
- the prover fills it with one or more canonical serialized PoRep units;
- the verifier checks the inner Filecoin proof;
- the verifier challenges random leaves of an outer Merkle commitment over the region bytes;
- the verifier accepts only if the prover answers correctly and within the deadline.

This repository only claims erasure for **user-space accessible** memory in the challenged regions. It does **not** claim kernel, firmware, or device-wide erasure.

## Status

This repository is specified to be built in phases:

- **Phase 0:** vendor upstream Filecoin Rust + bridge it into Python
- **Phase 1:** host-memory PoSE
- **Phase 2:** single-H100 HBM PoSE
- **Phase 3:** single-H100 hybrid host + HBM
- **Phase 4:** 8×H100 scale-out
- **Phase 5:** parity-gated Python Filecoin port promotion

See `docs/repository-spec.md` for the full normative spec.

## Repository shape

```text
repo/
  docs/
  vendor/rust-fil-proofs/
  rust/pose_filecoin_bridge/
  src/pose/
  bench_profiles/
  tests/
```

### Important directories

- `vendor/rust-fil-proofs/`
  - vendored upstream Filecoin Rust workspace snapshot
- `rust/pose_filecoin_bridge/`
  - thin Rust bridge exposing the official seal/verify path to Python
- `src/pose/prover/`
  - prover service, region materialization, outer proof response
- `src/pose/verifier/`
  - verifier service, leases, deadlines, result writing
- `src/pose/filecoin/mirror/`
  - deterministic Python mirrors used for parity testing
- `tests/parity/`
  - Python-vs-Rust equivalence tests
- `bench_profiles/`
  - named benchmark profiles for single-H100 and 8×H100 runs

## Key design rules

### 1. Production mode uses real Filecoin code

The production inner proof path must use the vendored upstream implementation.  
No fake or synthetic proof path is allowed in production mode.

### 2. Prover and verifier stay separate

The repository is intentionally split into:

- a prover process that generates and stores the object; and
- a verifier process that owns policy, challenges, deadlines, and verdicts.

### 3. Coverage is reported honestly

The output must always distinguish:

- total detected memory,
- usable memory,
- challenged/covered memory,
- real-PoRep bytes,
- alignment tail filler bytes.

### 4. Python ports require parity before promotion

The repository may progressively port deterministic Filecoin components into Python, but no component may replace the reference path without passing parity tests.

## CLI

### Prover

```bash
pose prover serve --config prover.toml
pose prover inspect
pose prover self-test
```

### Verifier

```bash
pose verifier run --profile single-h100-hybrid-max
pose verifier run --plan plan.yaml --json
pose verifier rechallenge --session-id <id>
pose verifier verify-record result.json
```

### Benchmarking

```bash
pose bench run --profile single-h100-hbm-max
pose bench matrix --profiles bench_profiles/
pose bench summarize results/*.json
```

## Expected output

Every verifier run and benchmark run should emit machine-readable JSON containing at least:

- session success/failure
- verdict
- coverage bytes
- real-PoRep ratio
- inner proof verification result
- outer proof verification result
- deadline and response timing
- per-phase timings
- environment metadata

The human-readable summary should surface:

- success
- covered host bytes
- covered HBM bytes per device
- real-PoRep ratio
- total runtime
- challenge-response runtime

## Benchmark profiles

The repository is expected to ship these profiles:

- `dev-small`
- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`
- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

These profiles should differ in:

- target devices
- reserve policy
- PoRep unit storage profile
- outer proof leaf size
- challenge count / soundness policy
- deadline policy
- repetition count

## Build philosophy

The default build should favor:

- reproducibility,
- visibility into upstream provenance,
- clear failure modes,
- benchmark artifact capture.

Recommended integration stack:

- Python packaging via `pyproject.toml`
- Rust bridge via `pyo3` + `maturin`
- pinned Python dependencies
- pinned Cargo lockfile

## Developer workflow

### Sync upstream Filecoin snapshot

```bash
make sync-upstream
```

### Build

```bash
make build
```

### Run tests

```bash
make test
make test-parity
```

### Run hardware tests

```bash
make test-hardware
```

### Run a benchmark profile

```bash
make bench PROFILE=single-h100-hbm-max
```

## Test strategy

The test suite is a first-class feature of the repository.

It should include:

- **unit tests**
  - serialization, hashing, protocol helpers, planners
- **parity tests**
  - Python mirror vs vendored Rust outputs
- **integration tests**
  - process boundaries, message flow, result schema
- **e2e tests**
  - full prover/verifier runs
- **adversarial tests**
  - stale proof replay, partial overwrite, timeout, wrong bytes
- **hardware tests**
  - single-H100 and 8×H100 benchmark validation
- **performance regression tests**
  - timing trend detection

## What “real PoSE” means here

This repository does **not** treat “a valid Filecoin proof somewhere” as sufficient.

A successful PoSE must mean:

- the prover generated a **real Filecoin PoRep**;
- the prover wrote the corresponding PoRep object into the challenged memory region;
- the verifier checked the official inner proof;
- the verifier checked a timed outer proof over the challenged in-memory bytes.

That is the core claim of the project.

## Non-goals

This repository is not trying to be:

- a full Filecoin miner,
- a blockchain integration project,
- a kernel attestation framework,
- a firmware security project,
- a disk/SSD erasure tool.

It is specifically a **user-space memory PoSE** framework built around a real Filecoin PoRep object.

## Contributing

Contributions must preserve the following invariants:

- no silent fallback to fake proofs in production mode;
- no weakening of the verifier/prover separation;
- no unverifiable local re-implementations of Filecoin logic in the production path;
- no new protocol behavior without corresponding tests;
- no Python port promotion without parity evidence.

Before making major changes, read:

- `docs/repository-spec.md`
- `docs/architecture.md`
- `docs/protocol.md`
- `docs/threat-model.md`

## Immediate implementation order

The expected order of implementation is:

1. vendor upstream Filecoin Rust
2. build the thin Rust bridge
3. stand up the Python prover and verifier skeletons
4. implement host-memory leased-region PoSE
5. implement single-H100 HBM leased-region PoSE
6. implement hybrid sessions
7. scale to 8×H100
8. expand the Python parity mirror and staged port

## Bottom line

This repository should become a benchmarkable, auditable, Python-first PoSE system that proves storage of a **real Filecoin PoRep object** in challenged host memory and GPU HBM, starting on a single H100 and scaling to 8×H100 while keeping the security story explicit and testable.
