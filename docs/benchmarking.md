# Benchmarking

## Purpose

Benchmarking is a first-class requirement because the repository makes a
time-bounded security claim. Artifact capture and comparison are not optional.

## Benchmark Classes

The benchmark harness is required to support:

- `cold`
- `warm`
- `rechallenge`

The scaffolded benchmark profile files already encode these fields so later
phases can attach real measurements without changing profile names or structure.

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

## Artifact Expectations

Benchmark runs must eventually archive:

- result JSON;
- benchmark logs;
- environment metadata;
- upstream commit identity;
- toolchain versions;
- GPU inventory where applicable.

## Foundation Status

The repository currently provides:

- profile definitions under `bench_profiles/`;
- a Python loader for those profiles;
- a summarizer scaffold;
- a lab matrix script entrypoint.

The current executable benchmark path is the host-only `minimal` profile flow
used by `dev-small`. HBM and hybrid profile names remain shipped for later
phases, but `bench run` reports them as not yet executable.

Full benchmark execution and regression tracking land in later phases.
