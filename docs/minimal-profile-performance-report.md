# Minimal Profile Performance Report

## Scope

This report explains why the current PoSE minimal benchmark is slow.

- Profile: `dev-small`
- Date: `2026-03-21`
- Goal: understand the current latency, not propose or implement fixes
- Primary archive: `.pose/benchmarks/dev-small/20260321T040704Z/`

The short version is simple: the minimal benchmark is slow because it still runs a real Filecoin seal, and almost all of the wall time is spent inside `seal_commit_phase2` in the Rust bridge.

## Benchmark Configuration

The shipped minimal profile is:

- `name: dev-small`
- `target_devices.host: true`
- `target_devices.gpus: []`
- `reserve_policy.host_bytes: 4096`
- `porep_unit_profile: minimal`
- `leaf_size: 4096`
- `response_deadline_ms: 5000`
- `repetition_count: 1`

Source: `bench_profiles/dev-small.yaml`

That means the benchmark only covers one 4 KiB host region and one canonical PoRep unit. Even so, the current implementation still runs the full real Filecoin proof path for that one unit.

## Commands Run

The measurements in this report came from:

- `uv run pose bench run --profile dev-small`
- repeated `VerifierService().run_session(load_profile("dev-small"))`
- repeated `VendoredFilecoinReference().seal()` in the same Python process
- `time -p uv run pose bench run --profile dev-small`
- `nvidia-smi` sampling during a minimal benchmark run

## End-to-End Results

### Archived Benchmark Run

From `.pose/benchmarks/dev-small/20260321T040704Z/summary.json` and `run-001.result.json`:

| Stage | Time (ms) | Share of total |
| --- | ---: | ---: |
| `total` | 45571 | 100.0% |
| `data_generation` | 45217 | 99.2% |
| `seal_commit_phase2` | 45026 | 98.8% |
| `discover` | 209 | 0.46% |
| `seal_pre_commit_phase2` | 140 | 0.31% |
| `seal_commit_phase1` | 16 | 0.04% |
| `inner_verify` | 12 | 0.03% |
| `seal_pre_commit_phase1` | 8 | 0.02% |
| `cleanup` | 1 | 0.00% |
| `challenge_response` | 0 | less than 1 ms |
| `outer_verify` | 0 | less than 1 ms |
| `object_serialization` | 0 | less than 1 ms |
| `copy_to_host` | 0 | less than 1 ms |
| `outer_tree_build` | 0 | less than 1 ms |

Important detail: the `TimingTracker` stores `int((elapsed) * 1000)`, so any stage below 1 ms is reported as `0 ms`. Those stages are not literally free; they are just below the timer's resolution.

### Repeated End-to-End Runs

Repeated `VerifierService().run_session(...)` calls show the same shape:

| Run | Total (ms) | Data generation (ms) | `seal_commit_phase2` (ms) |
| --- | ---: | ---: | ---: |
| 1 | 47763 | 47408 | 47188 |
| 2 | 44128 | 43788 | 43598 |
| 3 | 48075 | 47737 | 47530 |
| Mean | 46655 | 46311 | 46105 |

The benchmark varies by a few seconds from run to run, but the dominant stage does not change. One real seal still consumes essentially all of the wall time.

### Direct Seal Timing

Repeated direct `VendoredFilecoinReference().seal()` calls in the same process:

| Run | Seal wall time (ms) | `seal_commit_phase2` (ms) |
| --- | ---: | ---: |
| 1 | 45887 | 45675 |
| 2 | 46101 | 46055 |
| 3 | 46346 | 46296 |
| Mean | 46111 | 46009 |

This is the most important measurement in the report:

- direct seal wall time is almost identical to end-to-end benchmark wall time
- repeated seals in the same process stay at about 46 seconds
- warming the Python process does not materially reduce the cost

That means the main problem is not benchmark harness overhead, gRPC startup, JSON, or Python orchestration. The native proof generation itself is expensive.

### Whole-Command CPU Time

`time -p uv run pose bench run --profile dev-small` reported:

- `real 45.30`
- `user 364.39`
- `sys 4.20`

`user` time being much larger than `real` time is consistent with heavy multithreaded native compute. The benchmark is not spending 45 seconds waiting on Python; it is burning substantial CPU time in the underlying proof library.

### GPU Observation

During a minimal benchmark run on a machine with an H100 installed, `nvidia-smi` sampling showed no sustained nonzero GPU utilization. Memory use sat near baseline and power stayed around idle-to-low-load levels.

This does not prove that the GPU is never touched, because short bursts can be missed by 1 second polling. It does support the simpler reading: the current minimal profile is effectively CPU-dominant on this machine.

## Timing Attribution By Code Path

### `discover`

`discover` is measured in `src/pose/verifier/grpc_host_session.py` around `start_ephemeral_prover_server(...)`.

That path:

- starts a new Python subprocess running `pose cli main prover grpc-serve`
- waits until the server answers `Discover`
- polls readiness every 100 ms

Relevant code:

- `src/pose/verifier/grpc_host_session.py:135-137`
- `src/pose/verifier/grpc_client.py:260-285`

Observed cost: about 209 ms.

Interpretation: this is a real fixed startup cost, but it is still less than half a percent of total runtime.

### `data_generation`

`data_generation` wraps both:

- `generate_inner_porep(...)`
- `materialize_region_payloads(...)`

Relevant code:

- `src/pose/verifier/grpc_host_session.py:142-149`

For the minimal profile, this stage dominates because the prover side calls `_materialize_locally(...)`, which calls `reference.seal(...)` once for the one planned sector:

- `src/pose/prover/host_worker.py:22-51`
- `src/pose/prover/host_worker.py:78-86`
- `src/pose/filecoin/reference.py:86-88`

That eventually enters the Rust bridge:

- `rust/pose_filecoin_bridge/src/lib.rs:282-417`

Inside that Rust path the benchmark performs:

- piece file creation
- staged sector creation
- sealed sector file creation
- cache directory creation
- `seal_pre_commit_phase1`
- `seal_pre_commit_phase2`
- `seal_commit_phase1`
- `seal_commit_phase2`
- `verify_seal`
- extra blob collection from the sealed sector and cache directory

Observed cost: about 44 to 48 seconds end to end, with `seal_commit_phase2` consuming almost all of it.

### `seal_pre_commit_phase1`

Relevant code:

- `rust/pose_filecoin_bridge/src/lib.rs:313-328`

Observed cost:

- 8 ms in the archived benchmark run
- 10 to 33 ms across repeated direct seals

Interpretation: small fixed native work plus temporary-file setup.

### `seal_pre_commit_phase2`

Relevant code:

- `rust/pose_filecoin_bridge/src/lib.rs:330-341`

Observed cost:

- 140 ms in the archived benchmark run
- 8 to 145 ms across repeated direct seals

Interpretation: this stage has a noticeable cold/warm effect, but even the cold case is still small compared with the total runtime. It is not the main reason the benchmark is slow.

### `seal_commit_phase1`

Relevant code:

- `rust/pose_filecoin_bridge/src/lib.rs:343-359`

Observed cost: about 15 to 17 ms.

Interpretation: small compared with the total runtime.

### `seal_commit_phase2`

Relevant code:

- `rust/pose_filecoin_bridge/src/lib.rs:361-367`

Observed cost:

- 45026 ms in the archived benchmark run
- 43598 to 47530 ms across repeated end-to-end runs
- 45675 to 46296 ms across repeated direct seals in one process

Interpretation:

- this is the bottleneck
- the benchmark is spending almost all of its time inside the real `filecoin_proofs::seal_commit_phase2(...)` call
- the cost survives repeated runs in the same process, so it is not mostly one-time Python or module initialization

This is the core reason the current minimal benchmark is slow.

### `object_serialization`

Relevant code:

- `src/pose/prover/host_worker.py:41-47`

Observed cost: below 1 ms in the benchmark timing output.

Interpretation: turning one seal artifact into one minimal PoRep unit is negligible at this size.

### `copy_to_host`

Relevant code:

- `src/pose/prover/host_worker.py:105-113`

Observed cost: below 1 ms.

Interpretation: writing one 4 KiB payload into the host lease is negligible.

### `outer_tree_build`

Relevant code:

- `src/pose/prover/host_worker.py:115-117`

Observed cost: below 1 ms.

Interpretation: the outer Merkle tree for a single 4 KiB leaf is negligible.

### `inner_verify`

Relevant code:

- `src/pose/verifier/grpc_host_session.py:166-170`
- `src/pose/filecoin/reference.py:90-94`
- `rust/pose_filecoin_bridge/src/lib.rs:419-441`

Observed cost:

- 12 ms in the archived benchmark run
- 2 to 3 ms for direct verify after a freshly generated seal

Interpretation: verifier-side proof checking is cheap relative to proof generation.

### `challenge_response`

Relevant code:

- `src/pose/verifier/grpc_host_session.py:188-196`
- `src/pose/prover/host_worker.py:148-180`

Observed cost: below 1 ms.

Interpretation: the minimal benchmark only has one leaf and one challenge, so building the opening is effectively free at this scale.

### `outer_verify`

Relevant code:

- `src/pose/verifier/grpc_host_session.py:198-217`

Observed cost: below 1 ms.

Interpretation: checking one manifest, one Merkle opening, and one deadline is negligible.

### `cleanup`

Relevant code:

- host-side cleanup is driven from `src/pose/verifier/grpc_host_session.py`
- zeroization work in the host worker is implemented in `src/pose/prover/host_worker.py`

Observed cost: 1 ms.

Interpretation: zeroizing and releasing a 4 KiB host mapping is negligible.

## Why The Minimal Benchmark Is Slow

### 1. "Minimal" still means "real Filecoin seal"

The minimal profile only shrinks the amount of memory reserved by PoSE. It does not replace the inner proof system with a toy operation. The current path still calls the real Rust bridge, which still calls the real `filecoin_proofs` seal pipeline for a 2 KiB sector shape.

The benchmark is therefore not measuring "4 KiB of host-memory work." It is measuring "one real Filecoin seal plus a tiny amount of PoSE plumbing."

### 2. One native proof stage dominates everything else

`seal_commit_phase2` alone accounts for about 98.8% of total end-to-end runtime in the archived benchmark and about 99.8% of direct seal wall time in the repeated direct-seal measurements.

That makes every other stage second-order:

- gRPC startup is a few hundred milliseconds
- pre-commit and commit phase 1 are tens of milliseconds
- serialization, host copy, outer challenge, and outer verify are below 1 ms at this scale

### 3. Python and gRPC overhead are not the main problem

The strongest evidence is the direct-seal measurement:

- mean direct `seal()` wall time: about 46.1 s
- mean end-to-end benchmark time: about 46.7 s

The gap between those numbers is small. The benchmark harness and session protocol add only a few hundred milliseconds. Optimizing Python alone cannot recover tens of seconds.

### 4. The expensive work appears CPU-dominant here

The `time -p` output shows about 364 seconds of user CPU time during a 45 second wall-clock run, which is consistent with multithreaded native compute. The H100 also showed no sustained nonzero utilization during 1 second polling.

The safest conclusion is:

- the current minimal benchmark is bottlenecked by native proof generation
- on this machine, that bottleneck does not present as sustained GPU work

## Secondary Observations

### The bridge overhead outside the recorded inner phases is small

Comparing direct seal wall time with the sum of the bridge's inner timing map leaves only about 10 to 16 ms of unaccounted work per seal. That remaining time covers items such as:

- JSON marshaling over the Python bridge boundary
- temporary-file bookkeeping
- extra blob collection after sealing

Those costs are real, but they are tiny compared with a 45 to 46 second `seal_commit_phase2`.

### Cold effects exist, but they do not change the conclusion

`seal_pre_commit_phase2` showed a cold-to-warm drop from about 145 ms to about 8 to 9 ms in repeated direct seals. That confirms there is some setup or cache effect in earlier phases.

It does not materially change total wall time because `seal_commit_phase2` remains about 46 seconds on every run.

## Bottom Line

The current minimal benchmark is slow for one reason above all others: it spends nearly all of its wall time generating a real Filecoin proof in `seal_commit_phase2`.

At the current scale:

- PoSE orchestration costs are small
- host memory copying is small
- outer challenge/verification costs are small
- verifier-side inner proof checking is small
- the native proof-generation step dominates the benchmark

If optimization work starts from the minimal benchmark, the first-order target is the real inner proof path, not the surrounding Python session machinery.

## High-Value Follow-Up Profiling

If we continue the investigation, the next profiling passes should focus on the native proof stage itself:

- isolate `seal_commit_phase2` with Rust-native profiling or flamegraphs
- determine whether parameter loading, CPU thread count, or proof synthesis dominate inside that stage
- compare the current minimal bridge path against any alternative inner-proof strategy before spending time on Python-level cleanup
