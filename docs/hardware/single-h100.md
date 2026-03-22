# Single H100 Hardware Notes

## Goal

The first hardware target is a same-host deployment with:

- substantial user-space host memory coverage; and
- one H100 GPU for HBM coverage.

## Priority Order

Implementation order on this hardware is:

1. host-only PoSE;
2. HBM-only PoSE;
3. hybrid host plus HBM sessions.

## Profile Expectations

The foundation ships these single-H100 profiles:

- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`

The single-gpu HBM profile now measures available HBM at runtime and plans
toward the configured target fill ratio without an artificial 1 MiB cap.
Benchmark runs archive result JSON, summary metrics, logs, environment
metadata, toolchain information, and GPU inventory under
`.pose/benchmarks/` whenever `pose bench run --profile single-h100-hbm-max`
executes on target hardware.
