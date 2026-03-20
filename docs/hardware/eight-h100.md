# Eight H100 Hardware Notes

## Goal

The scale target is an eight-GPU H100 host that can run:

- HBM-only sessions across multiple devices; and
- hybrid host plus HBM sessions with per-device accounting.

## Required Additions Beyond Single-H100

The 8xH100 phase requires:

- a per-device worker model;
- region aggregation across devices;
- lab automation for repeatable runs;
- archived benchmark metadata for each job.

## Profiles

The foundation ships these scale-out profiles:

- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

These files are structural placeholders until real device measurements land.

