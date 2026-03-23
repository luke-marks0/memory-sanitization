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

The repository also ships `single-h100-hbm-small` as a development-only HBM
smoke profile. It uses `process_budget_dev`, keeps the targeted GPU visible,
and lowers the reported attacker budget for practical local regression. It is
not a production or theorem-level substitute for `single-h100-hbm-max`.

The repository also ships `single-h100-hybrid-small` as a development-only
hybrid host plus HBM smoke profile. It uses `process_budget_dev` to bound the
host-side attacker budget for local testing while keeping the targeted GPU
visible. It is not a production or theorem-level substitute for
`single-h100-hybrid-max`.

The single-gpu HBM profile now measures available HBM at runtime and plans
toward the configured target fill ratio without an artificial 1 MiB cap.
Benchmark runs archive result JSON, summary metrics, logs, environment
metadata, toolchain information, and GPU inventory under
`.pose/benchmarks/` whenever `pose bench run --profile single-h100-hbm-max`
executes on target hardware.
