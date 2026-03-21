# Minimal Profile Performance Report

## Scope

This report explains the current `dev-small` minimal benchmark after the bridge
was migrated to the upstream GPU backends, and compares it against the
pre-migration CPU-dominant baseline from the same repository.

- Profile: `dev-small`
- Update date: `2026-03-21`
- Current archive: `.pose/benchmarks/dev-small/20260321T052354Z/`
- Historical baseline archive: `.pose/benchmarks/dev-small/20260321T040704Z/`

The short version is:

- the GPU-backend migration materially improved the minimal benchmark;
- the H100 is now actually used during the run;
- the benchmark is still dominated by one real Filecoin proof, especially
  `seal_commit_phase2`.

## Benchmark Configuration

The shipped minimal profile is unchanged:

- `name: dev-small`
- `target_devices.host: true`
- `target_devices.gpus: []`
- `reserve_policy.host_bytes: 4096`
- `porep_unit_profile: minimal`
- `leaf_size: 4096`
- `response_deadline_ms: 5000`
- `repetition_count: 1`

Source: `bench_profiles/dev-small.yaml`

This is still a host-only PoSE benchmark over one 4 KiB region. The GPU does
not hold the challenged bytes for this profile. The GPU is only relevant because
the inner Filecoin proof path inside the Rust bridge can now offload proving
work to the upstream CUDA/OpenCL backends.

## Commands Run

The current measurements in this update came from:

- `uv run pose bench run --profile dev-small`
- `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader`

I left `BELLMAN_NO_GPU` unset for the benchmark so this exercised the normal
GPU-enabled bridge build.

## Archived Comparison

The table below compares the historical CPU-dominant run archived at
`.pose/benchmarks/dev-small/20260321T040704Z/` against the new GPU-enabled run
archived at `.pose/benchmarks/dev-small/20260321T052354Z/`.

| Metric | CPU-dominant baseline (ms) | GPU-enabled current (ms) | Change |
| --- | ---: | ---: | --- |
| `total` | 45571 | 12698 | 3.59x faster, 72.1% lower |
| `data_generation` | 45217 | 12138 | 3.73x faster, 73.2% lower |
| `seal_commit_phase2` | 45026 | 11834 | 3.81x faster, 73.7% lower |
| `seal_pre_commit_phase2` | 140 | 144 | effectively unchanged |
| `seal_commit_phase1` | 16 | 20 | effectively unchanged |
| `inner_verify` | 12 | 12 | unchanged |
| `discover` | 209 | 310 | 101 ms higher |

For the current run:

- verdict: `SUCCESS`
- `cpu_fallback.detected_run_count`: `0`
- `benchmark.log` recorded `cpu_fallback=false`

The fallback signal matters because the repo now records whether a run silently
dropped back to CPU. The current benchmark did not.

## Current Timing Shape

From `.pose/benchmarks/dev-small/20260321T052354Z/summary.json` and
`run-001.result.json`:

| Stage | Time (ms) | Share of total |
| --- | ---: | ---: |
| `total` | 12698 | 100.0% |
| `data_generation` | 12138 | 95.6% |
| `seal_commit_phase2` | 11834 | 93.2% |
| `discover` | 310 | 2.4% |
| `seal_pre_commit_phase2` | 144 | 1.1% |
| `seal_pre_commit_phase1` | 26 | 0.2% |
| `seal_commit_phase1` | 20 | 0.2% |
| `inner_verify` | 12 | 0.1% |
| `copy_to_host` | 1 | less than 0.1% |
| `cleanup` | 1 | less than 0.1% |
| `challenge_response` | 0 | less than 1 ms |
| `outer_verify` | 0 | less than 1 ms |
| `object_serialization` | 0 | less than 1 ms |
| `outer_tree_build` | 0 | less than 1 ms |

The timer still stores integer milliseconds, so any stage reported as `0 ms` is
below timer resolution rather than literally free.

## GPU Observation

During the GPU-enabled benchmark run, a spot `nvidia-smi` sample reported:

- device: `NVIDIA H100 80GB HBM3`
- utilization: `55 %`
- memory used: `2575 MiB`
- power draw: `172.54 W`

That is materially different from the earlier CPU-dominant behavior described in
the previous version of this report. The H100 is now doing real work during the
minimal benchmark.

This was only one sample, not a continuous profiler trace, so it should be read
as confirmation of GPU activity rather than a full utilization study.

## Interpretation

The migration to upstream GPU backends changed the minimal benchmark in an
important way, but it did not change what the benchmark fundamentally measures.

What changed:

- the end-to-end minimal benchmark dropped from about `45.6 s` to about
  `12.7 s`;
- the dominant proving stage, `seal_commit_phase2`, dropped from about
  `45.0 s` to about `11.8 s`;
- the archived run now explicitly reports that no CPU fallback occurred.

What did not change:

- the benchmark still spends almost all of its wall time generating one real
  Filecoin proof;
- the surrounding Python, gRPC, JSON, host-memory lease, and outer-PoSE work
  are still small compared with the native proving cost;
- `seal_commit_phase2` is still the main bottleneck by a wide margin.

Put differently: the old report’s conclusion that the H100 was effectively idle
is no longer true, but the broader conclusion that "the minimal benchmark is
mostly measuring the inner Filecoin proof path" is still true.

## Current Bottom Line

The current `dev-small` benchmark is no longer behaving like a CPU-only run.
The GPU-enabled bridge reduced end-to-end time by about `72%`, and the archived
result clearly shows no CPU fallback.

Even after that improvement, the benchmark is still dominated by
`seal_commit_phase2`. If further optimization work starts from the minimal
benchmark, the first-order target remains the internal proving path inside the
real Filecoin seal flow, not the surrounding PoSE orchestration.
