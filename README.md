# pose

Python-first proof of secure erasure over host memory and GPU HBM using the
graph-based PoSE-DB protocol from the bundled paper.

## Status

The repository is in active cutover to the PoSE-DB design defined in
`docs/repository-spec.md`.

Current repository policy:

- new work should target PoSE-DB only;
- the migration is a hard cutover, not a compatibility layer;
- remaining references to the older design should be treated as migration debt
  and removed.

Progress is tracked in `docs/migration-checklist.md`.

## Repository Shape

```text
repo/
  docs/
    references/
  proto/
  src/pose/
    cli/
    common/
    graphs/
    hashing/
    prover/
    verifier/
    benchmarks/
  bench_profiles/
  tests/
```

Important documents:

- `docs/repository-spec.md`
- `docs/architecture.md`
- `docs/protocol.md`
- `docs/security-model.md`
- `docs/graph-construction.md`
- `docs/benchmarking.md`

## Target Design

The target repository implements:

- verifier-owned host and HBM leases;
- implicit depth-robust graph construction;
- in-place label generation into leased regions;
- timed single-label challenge rounds;
- explicit calibration establishing `q < gamma`;
- honest attacker-budget and claim-scope reporting.

## CLI

Current CLI personas live under:

- `pose prover ...`
- `pose verifier ...`
- `pose bench ...`

The verifier target workflows are:

- `pose verifier run --profile ...`
- `pose verifier rechallenge --session-id ...`
- `pose verifier verify-record ...`
- `pose verifier calibrate --profile ...`

## Development

The repository is still mid-migration. Until the cutover is complete:

- expect some modules to remain transitional;
- use the repository spec, not older implementation assumptions, as the source
  of truth;
- keep changes aligned with the checklist order so old concepts are removed
  rather than preserved.
