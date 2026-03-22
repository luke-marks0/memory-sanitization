# Benchmarking

## Purpose

Benchmarking is part of the protocol, not a separate convenience feature.

The repository makes timed claims, so benchmark artifacts must preserve enough
information to justify:

- resident lookup timing;
- transport overhead;
- hash-throughput assumptions;
- calibrated `q`;
- the `q < gamma` margin for each production profile.

## Benchmark Classes

The benchmark harness supports these run classes:

- `cold`
- `warm`
- `rechallenge`

## Required Profiles

The required named profiles are:

- `dev-small`
- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`
- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

Each profile should define:

- target devices and reserve policy;
- `w_bits`;
- graph family and hash backend;
- adversary model;
- attacker-budget assumption;
- fixed `r` or target success bound;
- deadline policy;
- calibration policy;
- cleanup policy;
- repetition count.

## Required Artifacts

Archived benchmark bundles should include:

- result artifacts;
- benchmark summary;
- benchmark log;
- calibration evidence;
- environment snapshot;
- GPU inventory where applicable;
- native acceleration versions where applicable.

## Reported Metrics

Benchmark summaries should report at minimum:

- success rate;
- deadline miss rate;
- coverage fraction;
- slack bytes;
- `q`, `gamma`, and `q/gamma`;
- attacker-budget assumption;
- reported success bound;
- per-device HBM coverage;
- verifier CPU time;
- rechallenge performance.
