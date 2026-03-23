# pose

Python-first PoSE-DB over verifier-leased host memory and GPU HBM using the
graph-based PoSE-DB protocol.

## What This Repository Implements

The repository one protocol (graph-based PoSE-DB) with two phases:

1. initialization: derive and materialize the session label array into
   verifier-leased regions;
2. fast phase: run timed challenge-response rounds over resident labels.

The verifier owns session planning, region leasing, challenge generation,
deadline enforcement, soundness reporting, and verdict construction.

The prover owns graph traversal during materialization, in-place label
generation, timed label lookup during the fast phase, and cleanup of temporary
state.

The repository is Python-first in orchestration and reference semantics.
Optional native acceleration may be used for performance-critical paths, but it
must remain semantically identical to the Python implementation.

## Repository Shape

```text
repo/
  bench_profiles/
  docs/
    hardware/
    performance/
    references/
  native/
  proto/
    pose/v1/
  scripts/
  src/pose/
    benchmarks/
    cli/
    common/
    graphs/
    hashing/
    protocol/
    prover/
    verifier/
  tests/
```

## Core Concepts

- challenged state is the label array
  `sigma = l(o1) || l(o2) || ... || l(om)`;
- the verifier allocates challenged host and HBM regions, then leases them to
  the prover;
- each fast round samples one slot uniformly from `[0, m)`, checks the returned
  label bytes, and enforces deadline `Delta`;
- valid production profiles require conservative calibration establishing
  `q < gamma`;
- result artifacts report formal claims and operational claims separately.

## CLI

The main personas are:

- `pose prover ...`
- `pose verifier ...`
- `pose bench ...`

Common verifier workflows:

- `pose verifier run --profile dev-small`
- `pose verifier run --profile single-h100-host-max`
- `pose verifier rechallenge --session-id <id>`
- `pose verifier verify-record result.json`
- `pose verifier calibrate --profile dev-small`

Benchmark workflows:

- `pose bench run --profile dev-small`
- `pose bench matrix --profiles bench_profiles/`
- `pose bench summarize results/*.json`

## Development

Setup and validation:

```bash
uv sync --extra dev
make build
make test
make test-parity
make test-graphs
```

Useful commands:

```bash
make calibrate PROFILE=dev-small
make bench PROFILE=dev-small
```

Available named benchmark profiles live in `bench_profiles/`, including:

- `dev-small`
- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`
- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

## Documentation

Start here:

- `docs/repository-spec.md`
- `docs/architecture.md`
- `docs/protocol.md`
- `docs/security-model.md`
- `docs/threat-model.md`
- `docs/graph-construction.md`
- `docs/result-schema.md`
- `docs/benchmarking.md`

Hardware and performance references:

- `docs/hardware/single-h100.md`
- `docs/hardware/eight-h100.md`
- `docs/performance/optimization-roadmap.md`
- `docs/performance/optimization-log.md`
