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

Values in the profile files are provisional planning defaults until the real
planner and hardware measurements are implemented.

