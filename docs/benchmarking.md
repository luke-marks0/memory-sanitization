# Benchmarking

## Purpose

Benchmarking is a first-class requirement because the repository makes a
time-bounded security claim. Artifact capture and comparison are not optional.

## Benchmark Classes

The benchmark harness is required to support:

- `cold`
- `warm`
- `rechallenge`

These benchmark classes are modeled directly in the profile files and the
archive metadata emitted by `pose bench run`.

## Required Profiles

The repository ships the required named profiles:

- `dev-small`
- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`
- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

Each profile captures:

- target devices;
- reserve policy;
- host and per-GPU target fractions;
- PoRep unit storage profile;
- leaf size;
- challenge policy;
- deadline policy;
- cleanup policy;
- repetition count.

## Artifact Archive

`pose bench run --profile ...` archives benchmark bundles under
`.pose/benchmarks/<profile>/<timestamp>/`.

Each archived bundle includes:

- result JSON;
- benchmark summary JSON;
- benchmark logs;
- environment metadata;
- upstream commit identity;
- toolchain versions;
- GPU inventory where applicable.

The summary reports:

- success rate;
- deadline miss rate;
- mean / p50 / p95 / p99 per timing key;
- coverage fraction;
- real-PoRep ratio;
- per-device HBM coverage;
- verifier CPU time;
- rechallenge performance where applicable.

## Current Status

The repository currently provides:

- profile definitions under `bench_profiles/`;
- a Python loader for those profiles;
- archived benchmark execution for host-only and single-gpu HBM profiles;
- machine-readable benchmark summaries;
- a lab matrix script entrypoint.

The current executable benchmark paths are:

- the host-only `minimal` profile flow used by `dev-small`;
- the single-gpu HBM flow used by `single-h100-hbm-max`.

Hybrid and multi-GPU profile names remain shipped for later phases.
