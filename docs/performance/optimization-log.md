# PoSE Optimization Log

## Purpose

This log is the evidence trail for performance work. Every optimization must
have a benchmarked before-state and after-state. If there is no benchmark
evidence, there is no accepted speedup.

## Rules

1. Never compare results from different profiles as if they were the same test.
2. Never compare runs with different correctness behavior.
3. Record artifact paths so every claim can be traced back to archived JSON.
4. Record the tests or parity gates run with the change.
5. If a change speeds up one phase but regresses total runtime or correctness,
   record that explicitly and reject or revise it.
6. Smoke-scale profiles are diagnostic only. A win that is specific to
   smoke-scale sessions, or appears only on smoke-scale sessions, must not be
   accepted into default runtime selection unless it also improves
   representative larger-scale profiles for the same memory tier.

## Summary Script

Use:

```bash
scripts/perf_summary_table.sh \
  .pose/benchmarks/dev-small/20260323T024202Z/summary.json \
  .pose/benchmarks/single-h100-hbm-small/20260323T024202Z/summary.json \
  .pose/benchmarks/single-h100-hybrid-small/20260323T024241Z/summary.json
```

The script emits a markdown table from archived benchmark summaries.

## Entry 000: Baseline Before Optimization Work

- date: `2026-03-23`
- git head: `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- environment: Linux `6.5.0-45-generic`, Python `3.14.3`
- GPU: `NVIDIA H100 80GB HBM3`, driver `555.58.02`
- status: accepted as baseline

### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile dev-small
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hbm-small
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hybrid-small
```

### Artifacts

- `dev-small`: `.pose/benchmarks/dev-small/20260323T024202Z`
- `single-h100-hbm-small`: `.pose/benchmarks/single-h100-hbm-small/20260323T024202Z`
- `single-h100-hybrid-small`: `.pose/benchmarks/single-h100-hybrid-small/20260323T024241Z`

### Baseline Metrics

| profile | total ms | label_generation ms | expected_response_prep ms | graph_construction ms | fast_phase ms | verifier_cpu ms | q/gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev-small` | 28995 | 13298 | 12264 | 1601 | 206 | 14150 | 0.2163 |
| `single-h100-hbm-small` | 29693 | 13778 | 12339 | 1489 | 207 | 14120 | 0.4697 |
| `single-h100-hybrid-small` | 30851 | 13592 | 13647 | 1673 | 205 | 15674 | 0.3730 |

### Baseline Findings

- `label_generation` plus `expected_response_prep` accounts for about `88%` of
  total runtime on all three smoke profiles.
- `graph_construction` is consistently about `5%` of total runtime.
- `fast_phase_total` is under `1%` of total runtime on these profiles.
- The current optimization priority should therefore be:
  full-graph compute first, graph reuse second, transport last.

### Additional Structural Evidence

Challenge-closure check for the 4096-label smoke profiles:

- total graph nodes: `557054`
- ancestor closure for 64 sampled challenges: `549982`
- closure fraction: `98.7305%`

Implication:

- a verifier-only "compute just the challenged labels" strategy is unlikely to
  materially reduce work on the current graph family and challenge count.

## Entry 001: Bounded Graph Topology Cache

- date: `2026-03-23`
- git head: `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status: accepted
- hypothesis: reusing immutable `PoseDbGraph` objects by descriptor digest will
  reduce repeated verifier-side graph construction cost in multi-run,
  same-process workloads without changing any PoSE semantics.

### Change Scope

- files:
  `src/pose/graphs/construction.py`,
  `src/pose/graphs/__init__.py`,
  `tests/unit/test_pose_graphs.py`
- profiles benchmarked:
  temporary in-process `dev-small-r3` workload with `repetition_count: 3`

### Commands

```bash
# before
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from textwrap import dedent
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file

with tempfile.TemporaryDirectory(prefix='pose-graph-cache-before-') as temp_dir:
    profile_path = Path(temp_dir) / 'dev-small-r3.yaml'
    profile_path.write_text(dedent('''
name: dev-small-r3
benchmark_class: cold
target_devices:
  host: true
  gpus: []
reserve_policy:
  host_bytes: 131072
  per_gpu_bytes: 0
host_target_fraction: 1.0
per_gpu_target_fraction: 0.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 16384
challenge_policy:
  rounds_r: 64
  target_success_bound: 1.0e-9
  sample_with_replacement: true
deadline_policy:
  response_deadline_us: 2500
  session_timeout_ms: 60000
calibration_policy:
  lookup_samples: 512
  hash_measurement_rounds: 3
  hashes_per_round: 4096
  transport_overhead_us: 100
  serialization_overhead_us: 50
  safety_margin_fraction: 0.25
cleanup_policy:
  zeroize: true
  verify_zeroization: false
repetition_count: 3
transport_mode: grpc
coverage_threshold: 1.0
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 4294967296
  require_no_visible_gpus: true
  memlock_max_bytes: 0
  file_size_max_bytes: 0
''').strip() + '\n', encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary}, sort_keys=True))
PY

# after
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from textwrap import dedent
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file
from pose.graphs import clear_pose_db_graph_cache, pose_db_graph_cache_info

clear_pose_db_graph_cache()
with tempfile.TemporaryDirectory(prefix='pose-graph-cache-after-') as temp_dir:
    profile_path = Path(temp_dir) / 'dev-small-r3.yaml'
    profile_path.write_text(dedent('''
name: dev-small-r3
benchmark_class: cold
target_devices:
  host: true
  gpus: []
reserve_policy:
  host_bytes: 131072
  per_gpu_bytes: 0
host_target_fraction: 1.0
per_gpu_target_fraction: 0.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 16384
challenge_policy:
  rounds_r: 64
  target_success_bound: 1.0e-9
  sample_with_replacement: true
deadline_policy:
  response_deadline_us: 2500
  session_timeout_ms: 60000
calibration_policy:
  lookup_samples: 512
  hash_measurement_rounds: 3
  hashes_per_round: 4096
  transport_overhead_us: 100
  serialization_overhead_us: 50
  safety_margin_fraction: 0.25
cleanup_policy:
  zeroize: true
  verify_zeroization: false
repetition_count: 3
transport_mode: grpc
coverage_threshold: 1.0
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 4294967296
  require_no_visible_gpus: true
  memlock_max_bytes: 0
  file_size_max_bytes: 0
''').strip() + '\n', encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    cache_info = pose_db_graph_cache_info()
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary, 'cache_info': cache_info._asdict()}, sort_keys=True))
PY
```

### Before Artifacts

- `.pose/benchmarks/dev-small-r3/20260323T031937Z`

### After Artifacts

- `.pose/benchmarks/dev-small-r3/20260323T032211Z`

### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 28915.67 | 27612.33 | -1303.33 (-4.51%) |
| label_generation ms | 13134.67 | 12975.00 | -159.67 (-1.22%) |
| expected_response_prep ms | 12634.67 | 12358.67 | -276.00 (-2.18%) |
| graph_construction ms | 1357.33 | 509.33 | -848.00 (-62.48%) |
| fast_phase ms | 205.33 | 205.67 | +0.33 (+0.16%) |
| verifier_cpu ms | 14237.67 | 13074.67 | -1163.00 (-8.17%) |

### Cache Evidence

- after-run cache stats: `misses=1`, `hits=2`, `currsize=1`, `maxsize=8`

Interpretation:

- the benchmark used three repeated cold runs in one interpreter;
- the first run paid the graph-build miss;
- the second and third runs reused the cached graph topology;
- the large reduction in `graph_construction` aligns with the observed cache
  hits.

### Correctness Gates

- tests:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_pose_graphs.py tests/unit/test_paper_conformance.py tests/unit/test_verifier_service.py tests/unit/test_grpc_pose_db_runtime.py tests/parity/test_reference_only_mode.py tests/adversarial/test_host_fast_phase_attacks.py`
- tests:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_rechallenge.py tests/unit/test_calibration.py tests/unit/test_phase1_spec_guards.py tests/integration/test_cli_smoke.py`
- parity checks:
  cache-specific unit tests assert identical descriptors reuse the same object
  and distinct descriptors do not alias cache entries
- hardware checks:
  not rerun for this change because the cache only reuses immutable in-process
  topology objects and does not alter transport, calibration accounting, or
  result semantics

### Decision

- accepted because:
  it preserves semantics, has direct cache-hit evidence, and materially reduces
  repeated in-process `graph_construction` cost
- rejected claims:
  this is not a general cold CLI speedup for one-shot runs; it primarily helps
  repeated same-process workflows such as multi-run benchmarks and future warm
  paths
- follow-up:
  accelerated labeling remains the next higher-value optimization target

## Entry 002: Session-Scoped Accelerated Labeling

- date: `2026-03-23`
- git head: `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status: accepted
- hypothesis: precomputing per-session label-encoding prefixes and hashing one
  compact payload per label will reduce both verifier
  `expected_response_prep` and prover `label_generation` without changing any
  PoSE semantics.

### Change Scope

- files:
  `src/pose/hashing/random_oracle.py`,
  `src/pose/hashing/encoding.py`,
  `src/pose/hashing/__init__.py`,
  `src/pose/graphs/labeling.py`,
  `src/pose/prover/grpc_service.py`,
  `tests/unit/test_pose_hashing.py`,
  `tests/unit/test_pose_graphs.py`,
  `tests/parity/test_accelerated_labeling.py`,
  `tests/unit/test_grpc_pose_db_runtime.py`,
  `tests/unit/test_rechallenge.py`,
  `tests/adversarial/test_host_fast_phase_attacks.py`
- profiles benchmarked:
  temporary in-process `dev-small-r3` workload with `repetition_count: 3`

### Commands

```bash
# before
# reuse the accepted post-cache artifact from Entry 001:
# .pose/benchmarks/dev-small-r3/20260323T032211Z

# after
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from textwrap import dedent
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file
from pose.graphs import clear_pose_db_graph_cache, pose_db_graph_cache_info

clear_pose_db_graph_cache()
with tempfile.TemporaryDirectory(prefix='pose-accelerated-labeling-after-') as temp_dir:
    profile_path = Path(temp_dir) / 'dev-small-r3.yaml'
    profile_path.write_text(dedent('''
name: dev-small-r3
benchmark_class: cold
target_devices:
  host: true
  gpus: []
reserve_policy:
  host_bytes: 131072
  per_gpu_bytes: 0
host_target_fraction: 1.0
per_gpu_target_fraction: 0.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 16384
challenge_policy:
  rounds_r: 64
  target_success_bound: 1.0e-9
  sample_with_replacement: true
deadline_policy:
  response_deadline_us: 2500
  session_timeout_ms: 60000
calibration_policy:
  lookup_samples: 512
  hash_measurement_rounds: 3
  hashes_per_round: 4096
  transport_overhead_us: 100
  serialization_overhead_us: 50
  safety_margin_fraction: 0.25
cleanup_policy:
  zeroize: true
  verify_zeroization: false
repetition_count: 3
transport_mode: grpc
coverage_threshold: 1.0
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 4294967296
  require_no_visible_gpus: true
  memlock_max_bytes: 0
  file_size_max_bytes: 0
''').strip() + '\n', encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    cache_info = pose_db_graph_cache_info()
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary, 'cache_info': cache_info._asdict()}, sort_keys=True))
PY
```

### Before Artifacts

- `.pose/benchmarks/dev-small-r3/20260323T032211Z`

### After Artifacts

- `.pose/benchmarks/dev-small-r3/20260323T034444Z`

### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 27612.33 | 5278.67 | -22333.67 (-80.88%) |
| label_generation ms | 12975.00 | 1825.67 | -11149.33 (-85.93%) |
| expected_response_prep ms | 12358.67 | 1075.00 | -11283.67 (-91.30%) |
| graph_construction ms | 509.33 | 503.33 | -6.00 (-1.18%) |
| fast_phase ms | 205.67 | 205.00 | -0.67 (-0.32%) |
| verifier_cpu ms | 13074.67 | 1771.33 | -11303.33 (-86.45%) |

### Additional Notes

- cache evidence remained effectively unchanged:
  `misses=1`, `hits=2`, `currsize=1`, `maxsize=8`
- the improvement is therefore not explained by extra graph-cache hits
- `q_bound` increased from `632.00` to `686.67`, but the run remained valid with
  `q/gamma = 0.3353 < 1.0`
- a first streaming-hash prototype was benchmarked during development and was
  slower; it was discarded and is not accepted as an optimization entry

### Correctness Gates

- tests:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_pose_hashing.py tests/unit/test_pose_graphs.py tests/parity/test_accelerated_labeling.py tests/parity/test_reference_only_mode.py`
- tests:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_grpc_pose_db_runtime.py tests/unit/test_rechallenge.py tests/adversarial/test_host_fast_phase_attacks.py tests/unit/test_verifier_service.py tests/integration/test_cli_smoke.py`
- tests:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_calibration.py tests/unit/test_phase1_spec_guards.py tests/unit/test_paper_conformance.py`
- parity checks:
  helper-level parity for streaming encodings and hash backends, engine-level
  parity for `compute_node_labels`, `compute_challenge_labels`, and
  `compute_label_array`, and end-to-end runtime/adversarial/rechallenge checks
  that compare accelerated prover/verifier behavior against
  `label_engine="reference"`
- hardware checks:
  not rerun for this change; the optimization only changes local label
  construction and prover materialization, not calibration accounting or device
  claim semantics

### Decision

- accepted because:
  the accepted session-scoped prefix-cache design preserves parity and reduces
  the dominant end-to-end compute phases by more than `80%` on the benchmarked
  workload
- rejected claims:
  no claim is made here about multi-process warm reuse, GPU-only production
  hardware behavior, or soundness changes; the only accepted claim is the
  measured speedup on the archived workload above
- follow-up:
  benchmark the same label-oracle context on the HBM-only and hybrid smoke
  profiles, then investigate whether the same prefix-caching approach can be
  pushed further into calibration-time hash measurements

## Entry 003: HBM And Hybrid Smoke Validation For Accelerated Labeling

- date: `2026-03-23`
- git head: `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status: accepted
- hypothesis: the accepted session-scoped accelerated labeling path should
  materially reduce end-to-end runtime on the single-H100 HBM-only and hybrid
  smoke profiles as well, without changing success status or claim semantics.

### Change Scope

- files:
  no new code change; this entry validates the already accepted accelerated
  labeling implementation from Entry 002 on real HBM-only and hybrid smoke
  profiles
- profiles benchmarked:
  `single-h100-hbm-small`, `single-h100-hybrid-small`

### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hbm-small
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hybrid-small
```

### Before Artifacts

- `single-h100-hbm-small`: `.pose/benchmarks/single-h100-hbm-small/20260323T024202Z`
- `single-h100-hybrid-small`: `.pose/benchmarks/single-h100-hybrid-small/20260323T024241Z`

### After Artifacts

- `single-h100-hbm-small`: `.pose/benchmarks/single-h100-hbm-small/20260323T040242Z`
- `single-h100-hybrid-small`: `.pose/benchmarks/single-h100-hybrid-small/20260323T040255Z`

### Metric Delta

`single-h100-hbm-small`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 29693.00 | 6645.00 | -23048.00 (-77.62%) |
| label_generation ms | 13778.00 | 2417.00 | -11361.00 (-82.46%) |
| expected_response_prep ms | 12339.00 | 971.00 | -11368.00 (-92.13%) |
| graph_construction ms | 1489.00 | 1164.00 | -325.00 (-21.83%) |
| fast_phase ms | 207.00 | 206.00 | -1.00 (-0.48%) |
| verifier_cpu ms | 14120.00 | 2326.00 | -11794.00 (-83.53%) |

`single-h100-hybrid-small`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 30851.00 | 8273.00 | -22578.00 (-73.18%) |
| label_generation ms | 13592.00 | 2844.00 | -10748.00 (-79.08%) |
| expected_response_prep ms | 13647.00 | 1278.00 | -12369.00 (-90.64%) |
| graph_construction ms | 1673.00 | 1641.00 | -32.00 (-1.91%) |
| fast_phase ms | 205.00 | 205.00 | 0.00 (0.00%) |
| verifier_cpu ms | 15674.00 | 3214.00 | -12460.00 (-79.49%) |

### Additional Notes

- both smoke profiles remained successful:
  `success_rate = 1.0 -> 1.0`
- the HBM-only smoke profile became more conservative but still valid:
  `q_bound = 962 -> 1776`, `q/gamma = 0.4697 -> 0.8672 < 1`
- the hybrid smoke profile stayed comfortably valid:
  `q_bound = 764 -> 802`, `q/gamma = 0.3730 -> 0.3916 < 1`
- the dominant gains again come from `label_generation` and
  `expected_response_prep`, not from `fast_phase_total`

### Correctness Gates

- tests:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_calibration.py tests/unit/test_pose_hashing.py tests/parity/test_accelerated_labeling.py tests/unit/test_verifier_service.py tests/unit/test_grpc_pose_db_runtime.py tests/unit/test_rechallenge.py tests/adversarial/test_host_fast_phase_attacks.py tests/integration/test_cli_smoke.py`
- parity checks:
  the accelerated prover/verifier paths remain checked against
  `label_engine="reference"` in the runtime, rechallenge, and adversarial
  slices
- hardware checks:
  both benchmarks were rerun on the real single H100 visible in this
  environment

### Decision

- accepted because:
  the already accepted accelerated-labeling implementation delivers the same
  large end-to-end win on HBM-only and hybrid smoke profiles while preserving
  successful execution and theorem-level claim discipline
- rejected claims:
  no claim is made here about `single-h100-hbm-max` or production host-tier
  confinement; this entry only covers the development smoke profiles
- follow-up:
  evaluate rejected calibration-time hash measurement work separately and keep
  it out of the accepted runtime path unless the affected profiles are retuned

## Entry 004: Calibration Hash Measurement Prefix Cache

- date: `2026-03-23`
- git head: `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status: rejected
- hypothesis: applying the same session-scoped prefix cache to calibration-time
  hash-throughput measurement would reduce calibration overhead without harming
  practical smoke-profile executability.

### Change Scope

- files:
  attempted in `src/pose/benchmarks/calibration.py`
- profiles benchmarked:
  calibration hash microbenchmark, `single-h100-hbm-small`,
  `single-h100-hybrid-small`

### Commands

```bash
# calibration hash microbenchmark
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

from pose.benchmarks.calibration import _measure_hash_evaluations_per_second
from pose.graphs import build_graph_descriptor
from pose.hashing import internal_label_bytes

measurement_rounds = 3
hashes_per_round = 4096
output_bytes = 32
hash_backend = 'blake3-xof'
descriptor = build_graph_descriptor(
    label_count_m=32768,
    hash_backend=hash_backend,
    label_width_bits=256,
    graph_family='pose-db-drg-v1',
)
graph_descriptor_digest = descriptor.digest
predecessor_labels = tuple(bytes([index + 1]) * output_bytes for index in range(2))
session_seed = bytes.fromhex('11' * 32)

def reference_measurement():
    elapsed_by_round = []
    best_rate = 0.0
    for round_index in range(measurement_rounds):
        started = perf_counter_ns()
        for node_index in range(hashes_per_round):
            internal_label_bytes(
                session_seed=session_seed,
                graph_descriptor_digest=graph_descriptor_digest,
                node_index=node_index + (round_index * hashes_per_round),
                predecessor_labels=predecessor_labels,
                hash_backend=hash_backend,
                output_bytes=output_bytes,
            )
        elapsed_ns = perf_counter_ns() - started
        elapsed_by_round.append(elapsed_ns)
        best_rate = max(best_rate, hashes_per_round / (elapsed_ns / 1_000_000_000))
    return {
        'measurement_rounds': measurement_rounds,
        'hashes_per_round': hashes_per_round,
        'round_elapsed_ns': elapsed_by_round,
        'mean_elapsed_ns': sum(elapsed_by_round) / len(elapsed_by_round),
        'best_rate_hashes_per_second': best_rate,
    }

def optimized_measurement():
    elapsed_by_round = []
    for _round_index in range(measurement_rounds):
        started = perf_counter_ns()
        _measure_hash_evaluations_per_second(
            hash_backend=hash_backend,
            output_bytes=output_bytes,
            graph_descriptor_digest=graph_descriptor_digest,
            measurement_rounds=1,
            hashes_per_round=hashes_per_round,
        )
        elapsed_by_round.append(perf_counter_ns() - started)
    best_rate = _measure_hash_evaluations_per_second(
        hash_backend=hash_backend,
        output_bytes=output_bytes,
        graph_descriptor_digest=graph_descriptor_digest,
        measurement_rounds=measurement_rounds,
        hashes_per_round=hashes_per_round,
    )
    return {
        'measurement_rounds': measurement_rounds,
        'hashes_per_round': hashes_per_round,
        'round_elapsed_ns': elapsed_by_round,
        'mean_elapsed_ns': sum(elapsed_by_round) / len(elapsed_by_round),
        'best_rate_hashes_per_second': best_rate,
    }

reference = reference_measurement()
optimized = optimized_measurement()
run_root = Path('.pose/benchmarks/calibration-hash-micro') / datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
run_root.mkdir(parents=True, exist_ok=False)
(run_root / 'reference.json').write_text(json.dumps(reference, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(run_root / 'optimized.json').write_text(json.dumps(optimized, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

# attempted smoke-profile validation under the calibration optimization
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hbm-small
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hybrid-small
```

### Before Artifacts

- calibration reference microbenchmark:
  `.pose/benchmarks/calibration-hash-micro/20260323T040107Z/reference.json`
- attempted HBM smoke run under the optimization:
  `.pose/benchmarks/single-h100-hbm-small/20260323T040113Z`
- attempted hybrid smoke run under the optimization:
  `.pose/benchmarks/single-h100-hybrid-small/20260323T040129Z`

### After Artifacts

- calibration optimized microbenchmark:
  `.pose/benchmarks/calibration-hash-micro/20260323T040107Z/optimized.json`

### Metric Delta

Calibration hash microbenchmark:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| mean elapsed ns | 24829965.00 | 8860146.33 | -15969818.67 (-64.32%) |
| best hash rate / s | 166011.34 | 465663.83 | +299652.49 (+180.50%) |

Attempted smoke-profile effect:

- `single-h100-hbm-small`: `q_bound = 1776 -> 2598`, `q/gamma = 0.8672 -> 1.2686`, verdict `SUCCESS -> CALIBRATION_INVALID`
- `single-h100-hybrid-small`: `q_bound = 802 -> 2603`, `q/gamma = 0.3916 -> 1.2710`, verdict `SUCCESS -> CALIBRATION_INVALID`

### Correctness Gates

- tests:
  the experimental branch passed focused correctness tests before smoke
  benchmarking, but the smoke-profile runs themselves invalidated because the
  more aggressive calibration raised `q_bound` above `gamma`
- parity checks:
  helper-level label parity remained intact; the rejection is about resulting
  calibration behavior, not byte mismatches
- hardware checks:
  both rejected smoke runs were exercised on the real single H100 visible in
  this environment

### Decision

- rejected because:
  although the calibration hash loop itself became much faster, the resulting
  `q_bound` increase invalidated both the HBM-only and hybrid smoke profiles in
  this repository configuration
- accepted finding:
  the prefix-cache idea is viable as a raw calibration hash micro-optimization,
  but it requires coordinated retuning of deadlines and reserves before it can
  be accepted into the repo
- follow-up:
  keep the calibration path on the previous slower but smoke-profile-compatible
  measurement for now; if we revisit this, the next step is a deliberate
  retuning exercise rather than a silent optimization merge

## Exploratory Note: Larger HBM Scaling

This is not an optimization before/after entry. It records the largest HBM-only
development profiles that were practical to run in the current turn and the
resulting scaling signal.

### Feasibility Boundary

- a true `1 GiB` challenged footprint is not feasible with the current graph
  family and pure-Python object model
- implied parameters:
  `m = 33,554,432`, `n = 24`, estimated graph nodes `= 20,266,876,926`
- that is not just slow; it is beyond what the current graph materialization
  strategy can reasonably hold or traverse in Python

### Feasible Larger Benchmarks Run

- `single-h100-hbm-1mib` artifact:
  `.pose/benchmarks/single-h100-hbm-1mib/20260323T045538Z`
- `single-h100-hbm-2mib` artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T045711Z`

| profile | challenged bytes | estimated nodes | total ms | label_generation ms | expected_response_prep ms | graph_construction ms | scratch_peak_bytes | q/gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single-h100-hbm-1mib` | 1,048,576 | 7,012,350 | 71,414 | 27,596 | 13,493 | 14,703 | 35,381,936 | 0.0575 |
| `single-h100-hbm-2mib` | 2,097,152 | 15,990,782 | 161,293 | 63,500 | 27,739 | 34,780 | 80,736,726 | 0.0214 |

### What The Large Runs Changed

- label work is still the main problem:
  `label_generation + expected_response_prep` is about `57%` of total at
  `2 MiB`
- graph construction is no longer secondary noise:
  it grew to about `22%` of total at `2 MiB`
- the current in-process graph cache does not help these one-shot runs enough;
  larger-profile work now needs a cross-process or compact graph strategy
- scratch stays bounded and honest, but it scales materially:
  about `35 MB` at `1 MiB` and about `81 MB` at `2 MiB`

## Entry 005: Formula-Driven Implicit Graph Topology

- date:
  2026-03-23
- git head:
  `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status:
  accepted
- hypothesis:
  replace the full explicit predecessor-table build with a formula-driven
  implicit graph engine that emits the same canonical predecessor rows on
  demand, so cold large-profile graph construction collapses without changing
  graph semantics

#### Change Scope

- files:
  `src/pose/graphs/construction.py`,
  `src/pose/graphs/labeling.py`,
  `src/pose/prover/grpc_service.py`,
  `tests/unit/test_pose_graphs.py`
- profiles benchmarked:
  direct constructor microbenchmark at `m = 65,536`,
  `single-h100-hbm-2mib`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_pose_graphs.py \
  tests/unit/test_paper_conformance.py \
  tests/unit/test_pose_hashing.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_rechallenge.py \
  tests/adversarial/test_host_fast_phase_attacks.py \
  tests/integration/test_cli_smoke.py \
  tests/unit/test_calibration.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_phase1_spec_guards.py \
  tests/adversarial/test_memory_accounting.py \
  tests/parity/test_reference_only_mode.py \
  tests/unit/test_slot_planning.py \
  tests/unit/test_gpu_lease.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

The constructor microbenchmark artifact was produced by a short inline Python
harness comparing the preserved explicit reference builder
`_build_pose_db_graph_uncached(...)` against `_build_pose_db_graph_formula(...)`
for the `single-h100-hbm-2mib` descriptor.

#### Before Artifacts

- direct explicit-builder baseline:
  preserved in code as `_build_pose_db_graph_uncached(...)`
- prior end-to-end artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T045711Z`

#### After Artifacts

- direct constructor microbenchmark:
  `.pose/benchmarks/graph-construction-micro/20260323T000000Z/summary.json`
- end-to-end HBM artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T061142Z`

#### Metric Delta

Direct constructor microbenchmark at `m = 65,536`:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| graph build mean ms | 33,147.65 | 19.97 | -99.94% |

End-to-end `single-h100-hbm-2mib`:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 161,293 | 120,149 | -25.51% |
| label_generation ms | 63,500 | 80,063 | +26.08% |
| expected_response_prep ms | 27,739 | 39,295 | +41.66% |
| graph_construction ms | 34,780 | 43 | -99.88% |
| fast_phase ms | 205 | 205 | +0.00% |
| verifier_cpu ms | 62,984 | 39,692 | -36.98% |
| q/gamma | 0.0214 | 0.0288 | +34.33% |

#### Correctness Gates

- tests:
  `154` focused tests passed across graph parity, hashing, runtime, rechallenge,
  CLI smoke, calibration, verifier, spec-guard, accounting, slot-planning, and
  GPU-lease slices
- parity checks:
  added explicit-builder parity for `m in {1, 2, 3, 5, 8, 17, 33, 64}` and kept
  the existing reference vectors plus accelerated-labeling parity suite green
- hardware checks:
  real H100 end-to-end `single-h100-hbm-2mib` benchmark remained `SUCCESS`
  with full `2,097,152` covered HBM bytes and valid `q < gamma`

#### Decision

- accepted because:
  it removes the large cold graph-construction bottleneck almost entirely while
  preserving exact predecessor rows and challenge ordering
- important tradeoff:
  the time did not disappear entirely; a meaningful portion moved into
  `label_generation` and verifier `expected_response_prep` because those paths
  now regenerate predecessor rows on demand in Python rather than iterating a
  resident tuple table
- follow-up:
  the next optimization should fuse or cache the formula-driven predecessor
  traversal inside label-generation and verifier-prep loops, so we keep the
  implicit graph representation without paying repeated Python regeneration
  costs

## Entry 006: Compact Spec Traversal For Implicit Graph Walks

- date:
  2026-03-23
- git head:
  `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status:
  accepted
- hypothesis:
  the first formula-driven implementation still paid too much per-node Python
  overhead because it rebuilt predecessor tuples and, on some paths, rebuilt a
  full transient row list for each traversal; moving hot paths to streaming
  predecessor specs should improve verifier prep and modestly improve total
  runtime without changing graph semantics

#### Change Scope

- files:
  `src/pose/graphs/construction.py`,
  `src/pose/graphs/labeling.py`,
  `src/pose/prover/grpc_service.py`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_pose_graphs.py \
  tests/unit/test_pose_hashing.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_calibration.py \
  tests/unit/test_rechallenge.py \
  tests/unit/test_verifier_service.py \
  tests/integration/test_cli_smoke.py \
  tests/adversarial/test_host_fast_phase_attacks.py \
  tests/adversarial/test_memory_accounting.py \
  tests/unit/test_phase1_spec_guards.py \
  tests/unit/test_paper_conformance.py \
  tests/parity/test_reference_only_mode.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- prior accepted formula-driven artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T061142Z`

#### After Artifacts

- exploratory post-streaming artifacts before final prover-side tightening:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T064049Z`
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T064518Z`
- accepted post-tightening artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T064838Z`

#### Metric Delta

Accepted comparison: `20260323T061142Z -> 20260323T064838Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 120,149 | 116,078 | -3.39% |
| label_generation ms | 80,063 | 80,653 | +0.74% |
| expected_response_prep ms | 39,295 | 34,730 | -11.62% |
| graph_construction ms | 43 | 41 | -4.65% |
| fast_phase ms | 205 | 205 | +0.00% |
| verifier_cpu ms | 39,692 | 35,132 | -11.49% |
| q/gamma | 0.02878 | 0.02579 | -10.39% |

#### Correctness Gates

- tests:
  `223` focused tests passed across graph parity, hashing, accelerated labeling,
  runtime, verifier service, rechallenge, calibration, CLI smoke,
  adversarial host fast-phase, accounting, spec guards, and paper conformance
- parity checks:
  canonical predecessor rows and challenge ordering remained identical to the
  preserved explicit reference builder
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with full
  `2,097,152` covered HBM bytes and valid `q < gamma`

#### Decision

- accepted because:
  compact predecessor specs plus direct streaming into verifier/prover hot paths
  improved verifier prep materially and produced a modest end-to-end win while
  preserving exact graph semantics
- important nuance:
  the gain is verifier-heavy; prover `label_generation` is still essentially
  flat, so this is not the final answer for cold-session runtime
- follow-up:
  next work should target prover-side materialization overhead directly,
  especially predecessor-label loading, scratch bookkeeping, and any remaining
  Python object churn in `MaterializeLabels`

## Entry 007: Challenge Slot Tables In Prover Materialization

- date:
  2026-03-23
- git head:
  `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status:
  rejected
- hypothesis:
  precomputing per-challenge attachment and offset tables in the prover would
  reduce repeated `_resolve_slot(...)`, attachment lookup, and region-type
  dispatch overhead during `MaterializeLabels`

#### Change Scope

- files:
  `src/pose/prover/grpc_service.py`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_rechallenge.py \
  tests/adversarial/test_memory_accounting.py \
  tests/integration/test_cli_smoke.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- prior accepted implicit-graph artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T064838Z`

#### After Artifacts

- rejected slot-table artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T065649Z`

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 116,078 | 116,038 | -40 (-0.03%) |
| label_generation ms | 80,653 | 79,189 | -1,464 (-1.82%) |
| expected_response_prep ms | 34,730 | 36,138 | +1,408 (+4.05%) |
| graph_construction ms | 41 | 43 | +2 (+4.88%) |
| fast_phase ms | 205 | 206 | +1 (+0.49%) |
| verifier_cpu ms | 35,132 | 36,583 | +1,451 (+4.13%) |
| q/gamma | 0.02579 | 0.02127 | -0.00452 (-17.51%) |
| scratch_peak_bytes | 80,736,710 | 81,899,327 | +1,162,617 (+1.44%) |

#### Correctness Gates

- tests:
  `25` focused tests passed across runtime, verifier, rechallenge, memory
  accounting, and CLI smoke slices
- parity checks:
  no graph or label semantics changed; the optimization only altered prover
  bookkeeping around challenge-slot access
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with full
  `2,097,152` covered HBM bytes and valid `q < gamma`

#### Decision

- rejected because:
  the end-to-end result is effectively flat and not trustworthy as a real win,
  while verifier CPU and scratch usage regressed measurably
- implementation outcome:
  reverted from the codebase rather than keeping extra complexity without clear
  benchmark evidence
- follow-up:
  target per-label Python allocation and payload construction instead of adding
  more challenge-slot bookkeeping

## Entry 008: Reusable Fixed-Layout Hash Payloads

- date:
  2026-03-23
- git head:
  `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status:
  accepted
- hypothesis:
  reusing mutable fixed-layout source/internal hash payload templates inside
  `LabelOracleContext` will remove per-node payload rebuilding and some small
  object churn from both prover materialization and verifier expected-response
  preparation without changing any label bytes

#### Change Scope

- files:
  `src/pose/hashing/random_oracle.py`,
  `src/pose/graphs/labeling.py`,
  `src/pose/prover/grpc_service.py`,
  `tests/unit/test_pose_hashing.py`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_pose_hashing.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_rechallenge.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_calibration.py \
  tests/adversarial/test_memory_accounting.py \
  tests/integration/test_cli_smoke.py

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_pose_graphs.py \
  tests/unit/test_phase1_spec_guards.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_host_fast_phase_attacks.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- prior accepted implicit-graph artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T064838Z`

#### After Artifacts

- accepted reusable-payload artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T070418Z`

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 116,078 | 102,977 | -13,101 (-11.29%) |
| label_generation ms | 80,653 | 68,154 | -12,499 (-15.50%) |
| expected_response_prep ms | 34,730 | 34,136 | -594 (-1.71%) |
| graph_construction ms | 41 | 28 | -13 (-31.71%) |
| fast_phase ms | 205 | 205 | +0 (+0.00%) |
| verifier_cpu ms | 35,132 | 34,493 | -639 (-1.82%) |
| q/gamma | 0.02579 | 0.05649 | +0.03070 (+119.05%) |
| scratch_peak_bytes | 80,736,710 | 80,736,569 | -141 (-0.00%) |

#### Correctness Gates

- tests:
  `107` focused tests passed across hashing, accelerated-labeling parity,
  runtime, rechallenge, verifier, calibration, memory accounting, CLI smoke,
  graph parity, phase-guard, paper-conformance, and adversarial host
  fast-phase slices
- parity checks:
  added direct coverage that reusable source/internal payload templates emit
  the exact same bytes as the reference domain-separated label helpers for both
  `blake3-xof` and `shake256`
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with full
  `2,097,152` covered HBM bytes and valid `q < gamma`

#### Decision

- accepted because:
  it produced a clear double-digit end-to-end win and a large prover-side
  `label_generation` reduction without changing graph semantics, label bytes,
  or fast-phase behavior
- important nuance:
  `q/gamma` rose on this run due to calibration variability, but remained well
  below `1.0`, so the session stayed sound and valid
- follow-up:
  the next target should remove the remaining per-label hash setup and prover
  bookkeeping overhead, especially successor-count prepass, challenge-node
  lookup, and any remaining hash-prefix recomputation cost

## Entry 009: Preseeded Hash State Reuse Per Domain

- date:
  2026-03-23
- git head:
  `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status:
  rejected
- hypothesis:
  precomputing source/internal base hasher states and cloning them per label
  would avoid reabsorbing static prefix material on every label hash and speed
  up both prover materialization and verifier expected-response preparation

#### Change Scope

- files:
  temporary experiment in
  `src/pose/hashing/random_oracle.py`,
  `src/pose/hashing/blake3_backend.py`,
  `src/pose/hashing/shake256_backend.py`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_pose_hashing.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_rechallenge.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_calibration.py \
  tests/adversarial/test_memory_accounting.py \
  tests/integration/test_cli_smoke.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- prior accepted reusable-payload artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T070418Z`

#### After Artifacts

- rejected preseeded-hasher artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T071633Z`

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 102,977 | 118,353 | +15,376 (+14.93%) |
| label_generation ms | 68,154 | 76,568 | +8,414 (+12.35%) |
| expected_response_prep ms | 34,136 | 41,076 | +6,940 (+20.33%) |
| graph_construction ms | 28 | 57 | +29 (+103.57%) |
| fast_phase ms | 205 | 206 | +1 (+0.49%) |
| verifier_cpu ms | 34,493 | 41,528 | +7,035 (+20.40%) |
| q/gamma | 0.05649 | 0.02545 | -0.03104 (-54.94%) |

#### Correctness Gates

- tests:
  `51` focused tests passed across hashing, accelerated-labeling parity,
  runtime, rechallenge, verifier, calibration, accounting, and CLI smoke
- parity checks:
  the attempt preserved exact output bytes against the current helpers, so the
  rejection is purely performance-based
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with full
  `2,097,152` covered HBM bytes and valid `q < gamma`

#### Decision

- rejected because:
  backend `copy()` plus incremental `update(...)` overhead was more expensive
  than hashing the reusable fixed-layout payload directly in Python on this
  stack
- implementation outcome:
  reverted from the codebase after the benchmark
- follow-up:
  do not spend more time on pure-Python hash-state cloning unless a backend
  exposes a much cheaper copy/finalize path; the next optimization target
  should move back to prover bookkeeping or a native label engine

## Entry 010: Compact Successor Bookkeeping And Gated Verifier Streaming

- date:
  2026-03-23
- git head:
  `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- status:
  accepted
- hypothesis:
  compact successor bookkeeping should reduce prover-side materialization
  overhead and scratch pressure, while a streaming verifier challenge-label
  path should be available for very large profiles without forcing a slowdown
  on smaller profiles

#### Change Scope

- files:
  `src/pose/graphs/labeling.py`,
  `src/pose/prover/grpc_service.py`,
  `tests/parity/test_accelerated_labeling.py`
- profiles benchmarked:
  `single-h100-hbm-2mib`
- supporting hardware check:
  `single-h100-hbm-small`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_pose_hashing.py \
  tests/unit/test_pose_graphs.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_rechallenge.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_calibration.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_memory_accounting.py \
  tests/adversarial/test_host_fast_phase_attacks.py \
  tests/integration/test_cli_smoke.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hbm-small
```

#### Before Artifacts

- prior accepted large-profile artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T070418Z`

#### After Artifacts

- accepted large-profile artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T084206Z`
- supporting smoke artifact:
  `.pose/benchmarks/single-h100-hbm-small/20260323T084206Z`
- rejected intermediate always-on streaming artifacts:
  `.pose/benchmarks/single-h100-hbm-small/20260323T083536Z`
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T083846Z`

#### Metric Delta

Accepted comparison: `20260323T070418Z -> 20260323T084206Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 102,977 | 102,540 | -437 (-0.42%) |
| label_generation ms | 68,154 | 64,909 | -3,245 (-4.76%) |
| expected_response_prep ms | 34,136 | 36,782 | +2,646 (+7.75%) |
| graph_construction ms | 28 | 43 | +15 (+53.57%) |
| fast_phase ms | 205 | 207 | +2 (+0.98%) |
| verifier_cpu ms | 34,493 | 37,164 | +2,671 (+7.74%) |
| q/gamma | 0.05649 | 0.02832 | -0.02817 (-49.86%) |
| scratch_peak_bytes | 80,736,569 | 32,764,200 | -47,972,369 (-59.42%) |

#### Correctness Gates

- tests:
  `99` focused tests passed across hashing, graph parity,
  accelerated-labeling parity, runtime, rechallenge, verifier, calibration,
  paper-conformance, memory accounting, host fast-phase adversarial, and CLI
  smoke slices; a final fast slice after gating the streaming verifier path
  passed `73` more tests
- parity checks:
  added a forced-streaming parity test so the verifier challenge-label
  streaming path remains byte-identical to the preserved reference path even
  though it is not the default on current smaller profiles
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with full
  `2,097,152` covered HBM bytes and valid `q < gamma`;
  `single-h100-hbm-small` also remained `SUCCESS`

#### Decision

- accepted because:
  compact successor bookkeeping reduced prover `label_generation` and cut
  prover scratch by about `59%` on the large HBM profile, while preserving a
  small end-to-end win overall
- important nuance:
  an always-on verifier-streaming path was benchmarked first and was slower on
  current smoke and `2 MiB` profiles; the accepted implementation therefore
  keeps verifier streaming as a large-profile fallback that activates only when
  the full verifier label buffer would otherwise exceed `1 GiB`
- follow-up:
  the next optimization should target either a faster large-profile verifier
  streaming implementation, or a native label engine that can make both prover
  and verifier full-graph passes cheaper without the current Python overhead

## Entry 011: Rust Native Label Engine With Stage-Free Prover Streaming

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  accepted
- hypothesis:
  a coarse-grained Rust engine should beat the current Python fast paths if it
  owns formula-driven graph traversal and label hashing directly, while Python
  keeps only orchestration and attachment I/O; the prover path must avoid a
  full host-side challenge buffer so `declared_stage_copy_bytes` stays zero

#### Change Scope

- files:
  `native/pose_native_label_engine/Cargo.toml`,
  `native/pose_native_label_engine/src/lib.rs`,
  `src/pose/graphs/native_engine.py`,
  `src/pose/graphs/labeling.py`,
  `src/pose/graphs/__init__.py`,
  `src/pose/prover/grpc_service.py`,
  `src/pose/prover/service.py`,
  `src/pose/verifier/service.py`,
  `src/pose/verifier/rechallenge.py`,
  `tests/parity/test_native_labeling.py`,
  `scripts/build_native_label_engine.sh`,
  `pyproject.toml`
- profiles benchmarked:
  `single-h100-hbm-2mib`
- supporting hardware check:
  `single-h100-hbm-small`

#### Commands

```bash
scripts/build_native_label_engine.sh

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/parity/test_native_labeling.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_pose_graphs.py \
  tests/unit/test_pose_hashing.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_rechallenge.py \
  tests/unit/test_calibration.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_memory_accounting.py \
  tests/adversarial/test_host_fast_phase_attacks.py \
  tests/integration/test_cli_smoke.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile single-h100-hbm-small
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- prior accepted large-profile artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T084206Z`

#### After Artifacts

- initial native artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T090853Z`
- accepted refined large-profile artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T091111Z`
- supporting smoke artifact:
  `.pose/benchmarks/single-h100-hbm-small/20260323T090853Z`

#### Metric Delta

Accepted comparison: `20260323T084206Z -> 20260323T091111Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 102,540 | 10,077 | -92,463 (-90.17%) |
| label_generation ms | 64,909 | 5,211 | -59,698 (-91.97%) |
| expected_response_prep ms | 36,782 | 4,109 | -32,673 (-88.83%) |
| graph_construction ms | 43 | 26 | -17 (-39.53%) |
| fast_phase ms | 207 | 204 | -3 (-1.45%) |
| verifier_cpu ms | 37,164 | 4,329 | -32,835 (-88.35%) |
| q/gamma | 0.02832 | 0.04840 | +0.02008 (+70.91%) |
| scratch_peak_bytes | 32,764,200 | 82,444,933 | +49,680,733 (+151.63%) |

Refinement comparison inside the native engine: `20260323T090853Z -> 20260323T091111Z`

| metric | initial native | refined native | delta |
| --- | ---: | ---: | ---: |
| total ms | 10,592 | 10,077 | -515 (-4.86%) |
| label_generation ms | 5,770 | 5,211 | -559 (-9.69%) |
| expected_response_prep ms | 4,148 | 4,109 | -39 (-0.94%) |
| scratch_peak_bytes | 146,408,061 | 82,444,933 | -63,963,128 (-43.69%) |

#### Correctness Gates

- tests:
  `104` focused tests passed across native parity, accelerated parity, graph
  parity, hashing, runtime, verifier, rechallenge, calibration,
  paper-conformance, memory accounting, host fast-phase adversarial, and CLI
  smoke slices; a final native-specific sanity slice passed `17` more tests
- parity checks:
  added direct native-engine parity against the preserved reference path for
  both `blake3-xof` and `shake256`, and kept the runtime/verifier slices green
  with the native engine preferred at execution time
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with full
  `2,097,152` covered HBM bytes, `declared_stage_copy_bytes=0`, and valid
  `q < gamma`; `single-h100-hbm-small` also remained `SUCCESS`

#### Decision

- accepted because:
  the Rust + PyO3 architecture moved the hot path across a coarse boundary:
  formula-driven graph traversal and label hashing now happen in native code,
  while Python keeps only protocol orchestration and attachment writes
- architecture rationale:
  a stage-free prover callback path was chosen over returning a full challenge
  buffer to Python, because that would have introduced a host-side stage copy
  and changed the runtime accounting story; the accepted native prover path
  streams challenge labels directly into the existing attachment writes and
  keeps `declared_stage_copy_bytes=0`
- important nuance:
  the first accepted native draft used an `usize`-wide live-slot table and
  reached about `146 MB` scratch on the `2 MiB` profile; narrowing that table
  to `u32` kept the speed-up and cut native scratch by about `44%`
- follow-up:
  the next native-engine work should target scratch-shape reductions beyond the
  current `u32` slot table, plus a cleaner packaging story so the Rust module
  can be built or detected without manual environment setup

## Entry 012: Native In-Place Labeling For Host Destination Slots

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  accepted
- hypothesis:
  the native engine should be able to realize the paper-style in-place
  schedule for a contiguous host challenge array, using the destination slots
  themselves as workspace instead of maintaining a separate `O(m)` scratch
  label store; that should materially reduce prover scratch and may also speed
  up verifier challenge-array generation because the same in-place scheduler
  can replace the old native scratch-store path.

#### Change Scope

- files:
  `docs/repository-spec.md`,
  `native/pose_native_label_engine/src/lib.rs`,
  `src/pose/graphs/native_engine.py`,
  `src/pose/prover/grpc_service.py`,
  `tests/parity/test_native_labeling.py`
- profiles benchmarked:
  `single-host-2mib`
- supporting hardware check:
  `single-h100-hbm-2mib`

#### Commands

```bash
scripts/build_native_label_engine.sh

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/parity/test_native_labeling.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_pose_hashing.py \
  tests/unit/test_pose_graphs.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_rechallenge.py \
  tests/integration/test_cli_smoke.py \
  tests/unit/test_calibration.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_memory_accounting.py \
  tests/adversarial/test_host_fast_phase_attacks.py

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-host-2mib.yaml
PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- host baseline:
  `.pose/benchmarks/single-host-2mib/20260323T095353Z`
- HBM baseline:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T091111Z`

#### After Artifacts

- accepted host in-place artifact:
  `.pose/benchmarks/single-host-2mib/20260323T100342Z`
- supporting HBM artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T100342Z`

#### Metric Delta

Accepted comparison for the real in-place prover path: `single-host-2mib`,
`20260323T095353Z -> 20260323T100342Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 8,964 | 7,951 | -1,013 (-11.30%) |
| label_generation ms | 4,358 | 3,699 | -659 (-15.12%) |
| expected_response_prep ms | 4,094 | 3,769 | -325 (-7.94%) |
| graph_construction ms | 26 | 41 | +15 (+57.69%) |
| fast_phase ms | 205 | 206 | +1 (+0.49%) |
| verifier_cpu ms | 4,291 | 4,019 | -272 (-6.34%) |
| q/gamma | 0.04977 | 0.02936 | -0.02042 (-41.02%) |
| scratch_peak_bytes | 82,444,933 | 7,865,015 | -74,579,918 (-90.46%) |

Supporting comparison for HBM, where only verifier challenge-array generation
became in-place and the prover still uses the stage-free streaming callback:
`single-h100-hbm-2mib`, `20260323T091111Z -> 20260323T100342Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 10,077 | 10,003 | -74 (-0.73%) |
| label_generation ms | 5,211 | 5,461 | +250 (+4.80%) |
| expected_response_prep ms | 4,109 | 3,841 | -268 (-6.52%) |
| verifier_cpu ms | 4,329 | 4,114 | -215 (-4.97%) |
| scratch_peak_bytes | 82,444,933 | 82,444,933 | 0 (+0.00%) |

#### Correctness Gates

- tests:
  `106` focused tests passed across native parity, accelerated parity,
  hashing, graph parity, grpc runtime, verifier, rechallenge, CLI smoke,
  calibration, paper conformance, memory accounting, and host fast-phase
  adversarial slices
- parity checks:
  added a direct in-place host-fill parity test so the new native
  destination-slot path remains byte-identical to the preserved reference
  challenge array for both `blake3-xof` and `shake256`
- hardware checks:
  real host `single-host-2mib` remained `SUCCESS` with
  `declared_stage_copy_bytes=0`; real H100 `single-h100-hbm-2mib` also
  remained `SUCCESS`

#### Decision

- accepted because:
  contiguous host sessions now use the destination host slots themselves as
  the native workspace, which is both more faithful to the paper’s in-place
  schedule and much cheaper in scratch than the previous native scratch-store
  path
- architecture rationale:
  the accepted implementation reuses one recursive in-place scheduler in Rust
  for both verifier challenge-array construction and host-only prover
  materialization, while preserving the old stage-free callback path for HBM
  and mixed-region sessions that cannot yet realize the destination array
  directly from CPU-native code
- important nuance:
  this change does **not** make HBM materialization paper-faithful in-place;
  HBM still streams labels from CPU-native computation into GPU memory, so the
  main accepted benchmark for in-place labeling is the host-only profile
- follow-up:
  a faithful in-place HBM prover path will need direct device-side execution
  over the leased HBM slots rather than a host-side producer plus copy path

## Entry 013: CUDA HBM In-Place Prover Materialization

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  needs-follow-up
- hypothesis:
  a direct device-side in-place HBM materializer should eliminate the
  host-native producer plus copy path for contiguous GPU-only sessions,
  drastically reduce prover scratch, and remove `copy_to_hbm`; if the CUDA
  kernels and recursive schedule are efficient enough, it may also improve the
  end-to-end HBM profile.

#### Change Scope

- files:
  `native/pose_native_label_engine/Cargo.toml`,
  `native/pose_native_label_engine/build.rs`,
  `native/pose_native_label_engine/src/hbm_inplace.cu`,
  `native/pose_native_label_engine/src/lib.rs`,
  `src/pose/graphs/native_engine.py`,
  `src/pose/prover/grpc_service.py`,
  `tests/parity/test_native_labeling.py`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Commands

```bash
scripts/build_native_label_engine.sh

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/parity/test_native_labeling.py -q

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_grpc_pose_db_runtime.py -q

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_verifier_service.py \
  tests/unit/test_rechallenge.py \
  tests/integration/test_cli_smoke.py \
  tests/unit/test_calibration.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_memory_accounting.py \
  tests/adversarial/test_host_fast_phase_attacks.py -q

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- prior accepted HBM baseline:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T100342Z`

#### After Artifacts

- CUDA HBM in-place artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T105723Z`

#### Metric Delta

Comparison for real-H100 HBM prover materialization:
`single-h100-hbm-2mib`, `20260323T100342Z -> 20260323T105723Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 10,003 | 21,056 | +11,053 (+110.50%) |
| label_generation ms | 5,461 | 16,370 | +10,909 (+199.76%) |
| expected_response_prep ms | 3,841 | 3,992 | +151 (+3.93%) |
| graph_construction ms | 48 | 41 | -7 (-14.58%) |
| fast_phase ms | 205 | 206 | +1 (+0.49%) |
| verifier_cpu ms | 4,114 | 4,287 | +173 (+4.21%) |
| copy_to_hbm ms | 23 | 0 | -23 (-100.00%) |
| q/gamma | 0.02838 | 0.02832 | -0.00006 (-0.22%) |
| scratch_peak_bytes | 82,444,933 | 3,735,764 | -78,709,169 (-95.47%) |

#### Correctness Gates

- tests:
  the CUDA-enabled native engine rebuilt successfully; `56` focused tests
  passed across native parity, grpc runtime, verifier, rechallenge, CLI smoke,
  calibration, paper conformance, memory accounting, and host fast-phase
  adversarial slices
- parity checks:
  added a direct GPU parity test that fills a real leased HBM challenge array
  in place and verifies byte-identical output against the preserved reference
  challenge-label array
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with
  `declared_stage_copy_bytes=0` and `copy_to_hbm=0`

#### Decision

- accepted for architecture and compliance shape:
  contiguous GPU-only `blake3-xof` sessions now have a true device-side
  in-place prover path that writes directly into the leased HBM destination
  array instead of staging labels in host memory and then copying them to the
  device
- not accepted as a speed optimization yet:
  the first CUDA in-place implementation is substantially slower than the
  previous CPU-native producer path on the current `2 MiB` benchmark, even
  though it removes the copy step and dramatically cuts prover scratch
- architecture rationale:
  this is still the right structural direction for paper-faithful HBM
  materialization because it eliminates the separate host-side surrogate label
  store and uses the destination HBM slots themselves as workspace; the
  current issue is kernel/schedule efficiency, not model correctness
- follow-up:
  optimize the CUDA scheduler and kernels before relying on this path for
  performance claims; likely targets are kernel launch granularity, temporary
  device-buffer churn, and reducing repeated host-driven recursion overhead

## Entry 014: CUDA HBM In-Place Synchronization And Plan-Caching Round

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  accepted with follow-up
- hypothesis:
  the first CUDA in-place path is paying most of its penalty in host-driven
  synchronization and repeatedly copying the same merged-plan index tables to
  device memory; removing per-kernel `cudaDeviceSynchronize()` calls and
  caching per-dimension plan indices on the GPU should recover a large share of
  the lost runtime without giving up the direct-in-HBM in-place structure.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Commands

```bash
scripts/build_native_label_engine.sh

PYTHONPATH=src .venv/bin/python -m py_compile \
  src/pose/graphs/native_engine.py \
  src/pose/prover/grpc_service.py \
  tests/parity/test_native_labeling.py

PYTHONPATH=src .venv/bin/python -m pytest tests/parity/test_native_labeling.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_grpc_pose_db_runtime.py -q

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_verifier_service.py \
  tests/unit/test_rechallenge.py \
  tests/integration/test_cli_smoke.py \
  tests/unit/test_calibration.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_memory_accounting.py \
  tests/adversarial/test_host_fast_phase_attacks.py -q

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Before Artifacts

- first CUDA HBM in-place attempt:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T105723Z`

#### After Artifacts

- optimized CUDA HBM in-place artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T110723Z`

#### Metric Delta

Comparison for the first optimization round on the real H100:
`single-h100-hbm-2mib`, `20260323T105723Z -> 20260323T110723Z`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 21,056 | 14,080 | -6,976 (-33.13%) |
| label_generation ms | 16,370 | 9,829 | -6,541 (-39.96%) |
| expected_response_prep ms | 3,992 | 3,558 | -434 (-10.87%) |
| graph_construction ms | 41 | 41 | 0 (+0.00%) |
| fast_phase ms | 206 | 205 | -1 (-0.49%) |
| verifier_cpu ms | 4,287 | 3,833 | -454 (-10.59%) |
| q/gamma | 0.02832 | 0.02783 | -0.00049 (-1.72%) |
| scratch_peak_bytes | 3,735,764 | 7,340,252 | +3,604,488 (+96.49%) |

For context against the pre-CUDA HBM baseline
`20260323T100342Z -> 20260323T110723Z`, the optimized CUDA path is still
slower overall (`10,003 ms -> 14,080 ms`) but preserves the structural gains:
`copy_to_hbm = 0` instead of `23 ms`, and `scratch_peak_bytes` is still down
from `82,444,933` to `7,340,252`.

#### Correctness Gates

- tests:
  rebuild succeeded, `56` focused tests passed across native parity, grpc
  runtime, verifier, rechallenge, CLI smoke, calibration, paper conformance,
  memory accounting, and host fast-phase adversarial slices
- parity checks:
  the real-GPU parity test for direct HBM in-place fill remained green after
  the launch/synchronization changes
- hardware checks:
  real H100 `single-h100-hbm-2mib` remained `SUCCESS` with
  `declared_stage_copy_bytes=0`, `copy_to_hbm=0`, and `coverage_fraction=1.0`

#### Decision

- accepted because:
  this round recovered about a third of the initial CUDA regression while
  keeping the device-side in-place architecture intact
- important tradeoff:
  caching device-side merged plans increases scratch relative to the first CUDA
  attempt, but the path is still far below the old streaming HBM prover path
  and still eliminates the host-to-HBM copy step
- remaining gap:
  the optimized CUDA path is still slower than the previous CPU-native
  materializer on the `2 MiB` benchmark, so HBM in-place is now structurally
  correct and much less scratch-heavy, but not yet the fastest implementation
- follow-up:
  the next likely wins are reducing kernel launch count and host recursion
  overhead, not more synchronization trimming

## Entry 015: Cooperative Fused CUDA Connector And Merge Kernels

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  rejected
- hypothesis:
  replacing the many small connector and merge kernel launches with
  cooperative-grid fused kernels should reduce launch overhead and host
  recursion cost on the single-H100 `2 MiB` profile.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Before Artifacts

- prior accepted CUDA HBM in-place baseline:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T110723Z`

#### After Artifacts

- rejected cooperative-fused artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T111550Z`

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 14,080 | 16,755 | +2,675 (+19.00%) |
| label_generation ms | 9,829 | 12,249 | +2,420 (+24.62%) |
| expected_response_prep ms | 3,558 | 3,678 | +120 (+3.37%) |
| verifier_cpu ms | 3,833 | 3,963 | +130 (+3.39%) |
| scratch_peak_bytes | 7,340,252 | 7,340,252 | 0 (+0.00%) |

#### Correctness Gates

- tests:
  parity, grpc runtime, verifier, rechallenge, calibration, CLI smoke, paper
  conformance, memory accounting, and host fast-phase adversarial slices all
  stayed green during the experiment
- hardware checks:
  the real H100 run still returned `SUCCESS` and preserved
  `declared_stage_copy_bytes=0` and `copy_to_hbm=0`

#### Decision

- rejected because:
  the cooperative fused kernels were slower on the real H100 profile despite
  preserving correctness; the added grid-wide synchronization cost outweighed
  the launch-count reduction on this workload
- follow-up:
  the cooperative path was disabled again and not left active in the runtime

## Entry 016: Fixed-Layout CUDA Payload Prefixes

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  rejected
- hypothesis:
  mirroring the earlier fixed-layout payload win from the CPU-native engine
  inside the CUDA kernels should reduce per-label payload construction cost in
  the direct-in-HBM path.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`
- profiles benchmarked:
  `single-h100-hbm-2mib`

#### Before Artifacts

- prior accepted CUDA HBM in-place baseline:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T110723Z`

#### After Artifacts

- rejected fixed-layout payload artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T111932Z`
- restored-code validation artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z`

#### Metric Delta

Rejected experiment versus the accepted CUDA HBM baseline:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 14,080 | 15,589 | +1,509 (+10.72%) |
| label_generation ms | 9,829 | 10,846 | +1,017 (+10.35%) |
| expected_response_prep ms | 3,558 | 3,985 | +427 (+12.00%) |
| verifier_cpu ms | 3,833 | 4,292 | +459 (+11.97%) |
| scratch_peak_bytes | 7,340,252 | 7,341,060 | +808 (+0.01%) |

Post-revert validation for the current workspace:

| metric | restored current |
| --- | ---: |
| total ms | 14,973 |
| label_generation ms | 10,553 |
| expected_response_prep ms | 3,684 |
| verifier_cpu ms | 3,969 |
| scratch_peak_bytes | 7,340,252 |

#### Correctness Gates

- tests:
  parity, grpc runtime, verifier, rechallenge, calibration, CLI smoke, paper
  conformance, memory accounting, and host fast-phase adversarial slices all
  passed before and after the revert
- hardware checks:
  both the rejected run and the restored-code run remained `SUCCESS` on the
  real H100 with `copy_to_hbm=0`

#### Decision

- rejected because:
  the fixed-layout CUDA payload prefixes did not translate into a runtime win;
  they made the direct-in-HBM path slower on the target profile
- current state:
  the workspace was reverted to the prior non-cooperative CUDA HBM path and
  revalidated with the `20260323T112146Z` artifact

## Entry 017: CUDA HBM Plan-Compaction Round

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  rejected
- hypothesis:
  compressing and streamlining the CUDA merged-plan tables should lower
  `scratch_peak_bytes` and cut enough plan upload / index-lookup overhead to
  improve the HBM path on the repeated small H100 benchmark.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`
- profiles benchmarked:
  temporary `single-h100-hbm-small-r3` workload with `repetition_count: 3`

#### Commands

```bash
scripts/build_native_label_engine.sh

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/parity/test_native_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py -q

PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from textwrap import dedent
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file

profile_text = dedent('''
name: single-h100-hbm-small-r3
benchmark_class: cold
target_devices:
  host: false
  gpus: [0]
reserve_policy:
  host_bytes: 0
  per_gpu_bytes: 131072
host_target_fraction: 0.0
per_gpu_target_fraction: 1.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 65536
challenge_policy:
  rounds_r: 64
  target_success_bound: 1.0e-9
  sample_with_replacement: true
deadline_policy:
  response_deadline_us: 5000
  session_timeout_ms: 120000
calibration_policy:
  lookup_samples: 512
  hash_measurement_rounds: 3
  hashes_per_round: 4096
  transport_overhead_us: 180
  serialization_overhead_us: 90
  safety_margin_fraction: 0.25
cleanup_policy:
  zeroize: true
  verify_zeroization: false
repetition_count: 3
transport_mode: grpc
coverage_threshold: 1.0
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 17179869184
  require_no_visible_gpus: false
  memlock_max_bytes: 0
  file_size_max_bytes: 0
''').strip() + '\n'

with tempfile.TemporaryDirectory(prefix='pose-hbm-small-r3-') as temp_dir:
    profile_path = Path(temp_dir) / 'single-h100-hbm-small-r3.yaml'
    profile_path.write_text(profile_text, encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary}, sort_keys=True))
PY
```

#### Before Artifacts

- repeated small-HBM baseline:
  `.pose/benchmarks/single-h100-hbm-small-r3/20260323T113512Z`

#### After Artifacts

- rejected plan-compaction artifact:
  `.pose/benchmarks/single-h100-hbm-small-r3/20260323T114936Z`

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 1,833.00 | 1,820.67 | -12.33 (-0.67%) |
| label_generation ms | 918.33 | 925.00 | +6.67 (+0.73%) |
| expected_response_prep ms | 234.00 | 231.67 | -2.33 (-1.00%) |
| graph_construction ms | 0.67 | 0.67 | 0 (+0.00%) |
| fast_phase ms | 206.00 | 205.67 | -0.33 (-0.16%) |
| verifier_cpu ms | 490.33 | 466.00 | -24.33 (-4.96%) |
| scratch_peak_bytes | 327,900 | 127,184 | -200,716 (-61.21%) |

#### Correctness Gates

- tests:
  `tests/parity/test_native_labeling.py` and
  `tests/unit/test_grpc_pose_db_runtime.py` both remained green after the
  native rebuild
- parity checks:
  the direct GPU parity slice continued to pass while the experiment was live
- hardware checks:
  the repeated `single-h100-hbm-small-r3` benchmark remained `SUCCESS` on the
  H100 with `declared_stage_copy_bytes=0`

#### Decision

- rejected because:
  the HBM-path target metric regressed: `label_generation` got slower even
  though `scratch_peak_bytes` dropped sharply, so this round did not deliver a
  defensible speedup for the path being optimized
- follow-up:
  keep the scratch reduction evidence, but revert the code and look for a
  backend-selection or algorithmic win instead of plan-compaction alone

## Entry 018: Small-HBM Native Streaming Selector

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  reverted by policy
- hypothesis:
  the direct CUDA in-place HBM path has a fixed launch/setup cost that is too
  high for smoke-scale HBM sessions; routing only very small HBM sessions to
  the existing native streaming producer should improve runtime while keeping
  the streaming scratch bounded under a small explicit cap.

#### Change Scope

- files:
  `src/pose/prover/grpc_service.py`,
  `tests/unit/test_verifier_service.py`
- profiles benchmarked:
  temporary `single-h100-hbm-small-r3` workload with `repetition_count: 3`

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_verifier_service.py \
  tests/unit/test_grpc_pose_db_runtime.py -q

PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from textwrap import dedent
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file

profile_text = dedent('''
name: single-h100-hbm-small-r3
benchmark_class: cold
target_devices:
  host: false
  gpus: [0]
reserve_policy:
  host_bytes: 0
  per_gpu_bytes: 131072
host_target_fraction: 0.0
per_gpu_target_fraction: 1.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 65536
challenge_policy:
  rounds_r: 64
  target_success_bound: 1.0e-9
  sample_with_replacement: true
deadline_policy:
  response_deadline_us: 5000
  session_timeout_ms: 120000
calibration_policy:
  lookup_samples: 512
  hash_measurement_rounds: 3
  hashes_per_round: 4096
  transport_overhead_us: 180
  serialization_overhead_us: 90
  safety_margin_fraction: 0.25
cleanup_policy:
  zeroize: true
  verify_zeroization: false
repetition_count: 3
transport_mode: grpc
coverage_threshold: 1.0
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 17179869184
  require_no_visible_gpus: false
  memlock_max_bytes: 0
  file_size_max_bytes: 0
''').strip() + '\n'

with tempfile.TemporaryDirectory(prefix='pose-hbm-small-r3-stream-threshold-') as temp_dir:
    profile_path = Path(temp_dir) / 'single-h100-hbm-small-r3.yaml'
    profile_path.write_text(profile_text, encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary}, sort_keys=True))
PY
```

#### Before Artifacts

- repeated small-HBM baseline:
  `.pose/benchmarks/single-h100-hbm-small-r3/20260323T113512Z`

#### After Artifacts

- accepted small-HBM selector artifact:
  `.pose/benchmarks/single-h100-hbm-small-r3/20260323T115812Z`

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 1,833.00 | 1,525.67 | -307.33 (-16.77%) |
| label_generation ms | 918.33 | 686.33 | -232.00 (-25.26%) |
| expected_response_prep ms | 234.00 | 231.33 | -2.67 (-1.14%) |
| graph_construction ms | 0.67 | 0.67 | 0 (+0.00%) |
| fast_phase ms | 206.00 | 205.67 | -0.33 (-0.16%) |
| verifier_cpu ms | 490.33 | 462.00 | -28.33 (-5.78%) |
| scratch_peak_bytes | 327,900 | 2,941,573 | +2,613,673 (+796.49%) |

#### Correctness Gates

- tests:
  `tests/unit/test_verifier_service.py` and
  `tests/unit/test_grpc_pose_db_runtime.py` both passed with the selector in
  place
- parity checks:
  no label-generation algorithm changed; the selector only chooses between two
  already-parity-covered native backends
- hardware checks:
  the repeated `single-h100-hbm-small-r3` benchmark remained `SUCCESS` on the
  H100 with `declared_stage_copy_bytes=0`

#### Decision

- benchmark result:
  on the benchmarked small HBM workload, this selector delivered a real
  end-to-end speedup and a clear `label_generation` win by routing only
  smoke-scale HBM sessions to the faster native streaming producer
- reason reverted:
  this was a smoke-scale-only backend-selection optimization, and smoke-scale
  wins are now explicitly treated as diagnostic rather than sufficient for
  accepted runtime policy; the selector was therefore removed from the default
  HBM path even though the smoke benchmark itself was positive
- important tradeoff observed while the experiment was live:
  `scratch_peak_bytes` rose from about `0.31 MiB` to about `2.81 MiB`, which
  further reinforced that this was not the right general HBM direction
- follow-up:
  keep working on larger-session HBM optimizations that improve the direct CUDA
  in-place backend or otherwise translate to representative non-smoke HBM
  profiles

## Entry 019: Revert Smoke-Scale HBM Selector And Codify Benchmark Policy

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  accepted
- hypothesis:
  removing the smoke-scale-only HBM selector and codifying benchmark-acceptance
  policy in the spec and optimization log should keep the runtime focused on
  optimizations that matter for representative larger HBM sessions.

#### Change Scope

- files:
  `src/pose/prover/grpc_service.py`,
  `tests/unit/test_verifier_service.py`,
  `docs/repository-spec.md`,
  `docs/performance/optimization-log.md`
- profiles benchmarked:
  none; this round is a policy/documentation revert that restores the
  pre-Entry-018 HBM runtime selection behavior

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_verifier_service.py \
  tests/unit/test_grpc_pose_db_runtime.py -q

PYTHONPATH=src .venv/bin/python -m py_compile \
  src/pose/prover/grpc_service.py \
  tests/unit/test_verifier_service.py
```

#### Restored Runtime Reference

- restored HBM selector-free baseline:
  `.pose/benchmarks/single-h100-hbm-small-r3/20260323T113512Z`

#### Correctness Gates

- tests:
  `tests/unit/test_verifier_service.py` and
  `tests/unit/test_grpc_pose_db_runtime.py` both passed after removing the
  selector-specific logic and tests
- compile checks:
  `py_compile` passed for `src/pose/prover/grpc_service.py` and
  `tests/unit/test_verifier_service.py`
- validation scope:
  this round validates the code revert and documentation change; it does not
  claim a new performance result because the removed optimization was
  intentionally smoke-scale-specific

#### Decision

- accepted because:
  the runtime is back to the larger-session-oriented HBM backend policy, and
  both the repository spec and optimization log now state explicitly that
  smoke-scale profiles are diagnostic only
- policy statement:
  smoke-scale PoSE exists to help discover optimizations that later improve
  representative larger sessions; it is not itself a target for accepted
  runtime specialization

## Entry 020: Exact Native Scratch Breakdown Audit

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  accepted
- hypothesis:
  the current native scratch peaks should be decomposable exactly into a small
  number of accounted structures; if that is true, the next optimization
  target should be obvious and the log should record it explicitly instead of
  treating "scratch" as an opaque bucket.

#### Change Scope

- files:
  `scripts/profile_scratch_breakdown.py`,
  `docs/performance/optimization-log.md`
- profiles benchmarked:
  none; this round analyzes the current accepted host and HBM artifacts rather
  than changing runtime behavior

#### Commands

```bash
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/profile_scratch_breakdown.py

PYTHONPATH=src .venv/bin/python scripts/profile_scratch_breakdown.py \
  --mode host-in-place \
  --result-artifact .pose/benchmarks/single-host-2mib/20260323T100342Z/run-001.result.json

PYTHONPATH=src .venv/bin/python scripts/profile_scratch_breakdown.py \
  --mode cuda-hbm-in-place \
  --result-artifact .pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z/run-001.result.json
```

#### Artifacts Analyzed

- current host in-place artifact:
  `.pose/benchmarks/single-host-2mib/20260323T100342Z/run-001.result.json`
- current CUDA HBM in-place artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z/run-001.result.json`

#### Exact Breakdown

Current host native in-place scratch:

- covered bytes:
  `2,097,152`
- scratch peak bytes:
  `7,865,015`
- scratch / covered:
  `3.7503x`

| component | bytes | share of scratch |
| --- | ---: | ---: |
| merged_plan_index_tables | 7,864,328 | 99.9913% |
| label_oracle_payloads | 623 | 0.0079% |
| temp_label_buffers | 64 | 0.0008% |

Host-path details from the script:

- the reported scratch peak matched exactly
- the merged-plan tables cover dimensions `0..15`
- the three largest plan tables are dimension `15 = 4,194,304` bytes,
  dimension `14 = 1,966,080` bytes, and dimension `13 = 917,504` bytes;
  together they already account for about `89.99%` of the full scratch peak
- the non-plan residual is only `687` bytes total:
  `623` bytes of label-oracle payload templates plus two `32`-byte temp label
  buffers

Current CUDA HBM in-place scratch:

- covered bytes:
  `2,097,152`
- scratch peak bytes:
  `7,340,252`
- scratch / covered:
  `3.5001x`

| component | bytes | share of scratch |
| --- | ---: | ---: |
| host_merged_plan_cache | 3,670,024 | 49.9986% |
| device_merged_plan_cache | 3,670,024 | 49.9986% |
| pose_oracle_config | 204 | 0.0028% |

HBM-path details from the script:

- the reported scratch peak matched exactly
- the current recursion instantiates merged-plan dimensions `0..14`
- the same per-dimension plan tables are cached on both host and device, which
  is why the current HBM scratch is almost exactly a `2x` duplication of the
  per-side plan bytes plus a `204`-byte `PoseOracleConfig`
- the largest per-side plan tables are dimension `14 = 1,966,080` bytes,
  dimension `13 = 917,504` bytes, and dimension `12 = 425,984` bytes; across
  host and device together those three dimensions account for about `90.18%`
  of the full HBM scratch peak

#### Correctness Gates

- compile checks:
  `py_compile` passed for `scripts/profile_scratch_breakdown.py`
- accounting checks:
  the new script reproduced the reported `scratch_peak_bytes` exactly for both
  current artifacts
- claim-scope checks:
  both analyzed artifacts still report `declared_stage_copy_bytes=0`, so the
  extra scratch identified here is scheduler metadata and fixed configuration,
  not a surviving label shadow entering the fast phase

#### Decision

- accepted because:
  the current "extra scratch" story is now exact instead of approximate
- profiling conclusion:
  current host scratch is effectively all merged-plan index tables; current HBM
  scratch is those same tables duplicated across host and device plus a tiny
  fixed config block
- follow-up:
  meaningful scratch reduction now clearly means compressing, generating
  lazily, or eliminating merged-plan index tables; micro-optimizing hash
  payload templates or temp label buffers will not materially change the peak

## Entry 021: HBM Internal Profiling Counters And Direct Microbenchmark Harness

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  accepted
- hypothesis:
  the current direct CUDA HBM path needs internal observability before another
  optimization round; if the runtime can expose per-kernel launch counts,
  recursion counts, copy/sync counts, and a direct native microbenchmark, then
  future HBM changes can be judged on the real bottleneck instead of guesswork.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`,
  `native/pose_native_label_engine/src/lib.rs`,
  `src/pose/graphs/native_engine.py`,
  `src/pose/benchmarks/native_hbm_microbench.py`,
  `tests/unit/test_native_hbm_microbench.py`
- profiles benchmarked:
  direct native HBM microbenchmark, restored `single-h100-hbm-2mib`

#### Commands

```bash
scripts/build_native_label_engine.sh

PYTHONPATH=src .venv/bin/python -m py_compile \
  src/pose/graphs/native_engine.py \
  src/pose/benchmarks/native_hbm_microbench.py \
  tests/unit/test_native_hbm_microbench.py

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_native_hbm_microbench.py \
  tests/parity/test_native_labeling.py \
  tests/unit/test_grpc_pose_db_runtime.py -q

PYTHONPATH=src .venv/bin/python -m pose.benchmarks.native_hbm_microbench \
  --label-count-m 65536 \
  --graph-parameter-n 15 \
  --repetitions 1 \
  --output-json /tmp/native-hbm-microbench-restored.json

PYTHONPATH=src .venv/bin/python -m pose.cli.main bench run \
  --profile /tmp/single-h100-hbm-2mib.yaml
```

#### Artifacts

- direct native microbenchmark:
  `.pose/benchmarks/native-hbm-microbench/20260323T131307Z-restored.json`
- restored real-H100 HBM benchmark:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T131307Z`

#### Observed Metrics

Direct native microbenchmark on `m=65536`, `n=15`:

- `gpu_in_place.wall_ms = 21,629.65`
- `host_in_place.wall_ms = 3,675.32`
- `stream_gpu_writer.wall_ms = 4,920.58`
- `gpu_in_place.scratch_peak_bytes = 7,340,252`
- `gpu_in_place.total_kernel_launches = 655,016`
- `gpu_in_place.cooperative_launch_successes = 0`
- `gpu_in_place.cooperative_launch_fallbacks = 131,070`

Restored end-to-end HBM benchmark:

- `label_generation = 21,944 ms`
- `total = 26,974 ms`
- `scratch_peak_bytes = 7,340,252`
- `declared_stage_copy_bytes = 0`

#### Correctness Gates

- tests:
  `tests/unit/test_native_hbm_microbench.py`,
  `tests/parity/test_native_labeling.py`, and
  `tests/unit/test_grpc_pose_db_runtime.py` all passed after the native rebuild
- compile checks:
  `py_compile` passed for the new Python wrapper and benchmark harness modules
- hardware checks:
  the restored `single-h100-hbm-2mib` run remained `SUCCESS` with
  `declared_stage_copy_bytes=0`

#### Decision

- accepted because:
  this round adds observability without changing the default HBM runtime path
- profiling conclusion:
  the current direct CUDA HBM implementation is still dominated by a very large
  host-orchestrated kernel schedule, not by copy-to-HBM traffic or scratch
  spikes
- follow-up:
  future HBM optimization rounds should compare against the restored
  `20260323T131307Z` artifacts in the same measurement window rather than the
  older `20260323T110723Z` run, because the machine was materially slower later
  in the day even after the runtime was restored

## Entry 022: Small-Width Merged-Center Block-Fused Round

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  rejected and reverted
- hypothesis:
  replacing the many indexed merged-center layer launches on small-width HBM
  subproblems with one ordinary block-local fused kernel should reduce launch
  overhead enough to improve the direct CUDA path without increasing scratch.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`,
  `native/pose_native_label_engine/src/lib.rs`
- profiles benchmarked:
  direct native HBM microbenchmark, `single-h100-hbm-2mib`

#### Before Artifacts

- restored microbenchmark anchor:
  `.pose/benchmarks/native-hbm-microbench/20260323T131307Z-restored.json`
- restored end-to-end HBM anchor:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T131307Z`

#### After Artifacts

- rejected block-fused microbenchmark:
  `.pose/benchmarks/native-hbm-microbench/20260323T130300Z-round1.json`
- rejected block-fused HBM artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T130300Z`

#### Metric Delta

Compared against the restored current-window anchor:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| micro `gpu_in_place` ms | 21,629.65 | 22,301.13 | +671.49 (+3.10%) |
| micro `total_kernel_launches` | 655,016 | 279,306 | -375,710 (-57.36%) |
| label_generation ms | 21,944 | 22,476 | +532 (+2.42%) |
| total ms | 26,974 | 26,787 | -187 (-0.69%) |
| scratch_peak_bytes | 7,340,252 | 7,340,252 | 0 (+0.00%) |

#### Correctness Gates

- tests:
  the focused native benchmark, parity, and grpc runtime slices stayed green
  before the revert
- hardware checks:
  the real H100 run still returned `SUCCESS` with `declared_stage_copy_bytes=0`

#### Decision

- rejected because:
  the launch-count reduction did not translate into a speedup on the HBM-path
  target metric; `label_generation` and the direct microbenchmark both got
  slower even though the schedule launched far fewer kernels
- conclusion:
  a one-block fused merge kernel is not enough by itself; it removes launches
  but makes each small merged-center subproblem too expensive

## Entry 023: Direct Predecessor-Pointer CUDA Round

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  rejected and reverted
- hypothesis:
  removing explicit predecessor staging arrays from the active CUDA kernels
  should lower per-thread local-state pressure and recover runtime without
  changing the existing launch schedule or scratch shape.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`
- profiles benchmarked:
  direct native HBM microbenchmark, `single-h100-hbm-2mib`

#### Before Artifacts

- restored microbenchmark anchor:
  `.pose/benchmarks/native-hbm-microbench/20260323T131307Z-restored.json`
- restored end-to-end HBM anchor:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T131307Z`

#### After Artifacts

- rejected direct-pointer microbenchmark:
  `.pose/benchmarks/native-hbm-microbench/20260323T130658Z-round2.json`
- rejected direct-pointer HBM artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T130658Z`

#### Metric Delta

Compared against the restored current-window anchor:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| micro `gpu_in_place` ms | 21,629.65 | 21,857.56 | +227.91 (+1.05%) |
| micro `total_kernel_launches` | 655,016 | 655,016 | 0 (+0.00%) |
| label_generation ms | 21,944 | 22,128 | +184 (+0.84%) |
| total ms | 26,974 | 26,518 | -456 (-1.69%) |
| scratch_peak_bytes | 7,340,252 | 7,340,252 | 0 (+0.00%) |

#### Correctness Gates

- tests:
  the focused native benchmark, parity, and grpc runtime slices stayed green
  before the revert
- hardware checks:
  the real H100 run still returned `SUCCESS` with `declared_stage_copy_bytes=0`

#### Decision

- rejected because:
  the direct pointer change did not improve the HBM-path target metric; the
  direct microbenchmark and `label_generation` both remained slower than the
  restored current-window baseline
- conclusion:
  local predecessor staging is not the dominant problem in the current active
  kernels, or the direct-pointer version simply traded that pressure for more
  expensive memory reads

## Entry 024: Shared-Memory Small-Width Merged-Center Round

- date:
  2026-03-23
- git head:
  `39199965abc417b228298799ec2b0fd5e50dbb69`
- status:
  rejected and reverted
- hypothesis:
  if the small-width merged-center fusion keeps both labels and index tables in
  shared memory, then the HBM path might keep the launch-count win from Entry
  022 without paying the same per-subproblem global-memory cost.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`,
  `native/pose_native_label_engine/src/lib.rs`
- profiles benchmarked:
  direct native HBM microbenchmark, `single-h100-hbm-2mib`

#### Before Artifacts

- restored microbenchmark anchor:
  `.pose/benchmarks/native-hbm-microbench/20260323T131307Z-restored.json`
- restored end-to-end HBM anchor:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T131307Z`

#### After Artifacts

- rejected shared-memory microbenchmark:
  `.pose/benchmarks/native-hbm-microbench/20260323T131051Z-round3.json`
- rejected shared-memory HBM artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T131051Z`

#### Metric Delta

Compared against the restored current-window anchor:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| micro `gpu_in_place` ms | 21,629.65 | 22,766.36 | +1,136.71 (+5.26%) |
| micro `total_kernel_launches` | 655,016 | 279,306 | -375,710 (-57.36%) |
| label_generation ms | 21,944 | 23,246 | +1,302 (+5.93%) |
| total ms | 26,974 | 28,187 | +1,213 (+4.50%) |
| scratch_peak_bytes | 7,340,252 | 7,340,252 | 0 (+0.00%) |

#### Correctness Gates

- tests:
  the focused native benchmark, parity, and grpc runtime slices stayed green
  before the revert
- hardware checks:
  the real H100 run still returned `SUCCESS` with `declared_stage_copy_bytes=0`

#### Decision

- rejected because:
  keeping the small-width merged-center workspace in shared memory still did
  not beat the restored current-window baseline; it remained slower on both the
  direct microbenchmark and end-to-end `label_generation`
- conclusion:
  the direct CUDA HBM gap is not going to close through isolated small-width
  merge fusion alone; the next credible direction needs a different execution
  shape, most likely one that reduces host-driven scheduling without turning
  each tiny subproblem into a slower single-block kernel

## Entry 025: Arithmetic Host Merged-Index Schedule

- date:
  2026-03-23
- git head:
  `7d398b5df889bff3eb2413f4ca48827574df1049`
- status:
  accepted
- hypothesis:
  the host-side merged-center and ingress lookup tables can be replaced with an
  exact arithmetic recurrence, eliminating almost all native host scratch while
  preserving the paper-equivalent schedule and possibly reducing verifier-side
  prep time.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/lib.rs`,
  `tests/parity/test_native_labeling.py`,
  `scripts/profile_scratch_breakdown.py`
- profiles benchmarked:
  `single-host-2mib`, supporting rerun of `single-h100-hbm-2mib`

#### Commands

```bash
scripts/build_native_label_engine.sh

(cd native/pose_native_label_engine && cargo test -q)

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/parity/test_native_labeling.py \
  tests/parity/test_accelerated_labeling.py \
  tests/unit/test_pose_hashing.py \
  tests/unit/test_pose_graphs.py \
  tests/unit/test_grpc_pose_db_runtime.py \
  tests/unit/test_verifier_service.py \
  tests/unit/test_rechallenge.py \
  tests/integration/test_cli_smoke.py \
  tests/unit/test_calibration.py \
  tests/unit/test_paper_conformance.py \
  tests/adversarial/test_memory_accounting.py \
  tests/adversarial/test_host_fast_phase_attacks.py

PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
import yaml
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file

plan = load_json_file(Path('.pose/benchmarks/single-host-2mib/20260323T100342Z/plan.json'))
profile = plan['profile']
with tempfile.TemporaryDirectory(prefix='pose-host-2mib-arithmetic-') as temp_dir:
    profile_path = Path(temp_dir) / 'single-host-2mib.yaml'
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary}, sort_keys=True))
PY

PYTHONPATH=src .venv/bin/python scripts/profile_scratch_breakdown.py \
  --mode host-in-place-arithmetic \
  --result-artifact .pose/benchmarks/single-host-2mib/20260323T133119Z/run-001.result.json

PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
import yaml
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file

plan = load_json_file(Path('.pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z/plan.json'))
profile = plan['profile']
with tempfile.TemporaryDirectory(prefix='pose-h100-hbm-2mib-host-arithmetic-') as temp_dir:
    profile_path = Path(temp_dir) / 'single-h100-hbm-2mib.yaml'
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary}, sort_keys=True))
PY
```

#### Before Artifacts

- host table-backed anchor:
  `.pose/benchmarks/single-host-2mib/20260323T100342Z`
- current HBM anchor:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z`

#### After Artifacts

- host arithmetic artifact:
  `.pose/benchmarks/single-host-2mib/20260323T133119Z`
- supporting HBM rerun with host arithmetic verifier path:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T133150Z`

#### Metric Delta

Host path, compared against the table-backed anchor:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 7,951 | 7,571 | -380 (-4.78%) |
| label_generation ms | 3,699 | 3,496 | -203 (-5.49%) |
| expected_response_prep ms | 3,769 | 3,538 | -231 (-6.13%) |
| graph_construction ms | 41 | 41 | 0 (+0.00%) |
| fast_phase ms | 206 | 205 | -1 (-0.49%) |
| verifier_cpu ms | 4,019 | 3,801 | -218 (-5.42%) |
| scratch_peak_bytes | 7,865,015 | 687 | -7,864,328 (-99.99%) |

The new host scratch breakdown is exact:

- `label_oracle_payloads = 623` bytes
- `temp_label_buffers = 64` bytes
- `merged-plan scheduler metadata = 0` bytes

Supporting HBM rerun:

- `scratch_peak_bytes` stayed `7,340,252`
- `declared_stage_copy_bytes` stayed `0`
- `expected_response_prep` improved from `3,684` ms to `3,560` ms
- `verifier_cpu_time_ms` improved from `3,969` ms to `3,792` ms

#### Correctness Gates

- tests:
  `cargo test -q` in `native/pose_native_label_engine` passed, including a
  dimension-by-dimension arithmetic-vs-table index equivalence test through
  `d=10`
- parity checks:
  the focused Python parity, grpc runtime, soundness, and adversarial slices
  passed (`107 passed`), and the native host parity test now asserts
  `scratch_peak_bytes < 1024`
- hardware checks:
  both the host and supporting HBM reruns returned `SUCCESS` with
  `declared_stage_copy_bytes=0`

#### Decision

- accepted because:
  the arithmetic recurrence exactly preserved the merged schedule while
  removing the host-side `MergedPlan` cache entirely; host scratch fell from
  `7,865,015` bytes to `687` bytes and host timings improved instead of
  regressing
- conclusion:
  the large host scratch figure was almost entirely precomputed scheduler
  metadata, not label storage or hash workspace
- follow-up:
  the CUDA HBM path still carries its own host+device merged-plan caches, so
  the next step is to port the same arithmetic schedule into
  `native/pose_native_label_engine/src/hbm_inplace.cu`

## Entry 026: Arithmetic CUDA HBM Merged-Index Schedule

- date:
  2026-03-23
- git head:
  `7d398b5df889bff3eb2413f4ca48827574df1049`
- status:
  accepted
- hypothesis:
  the CUDA HBM merged-center path can use the same arithmetic ingress/center
  index recurrence as the host path, eliminating both the host and device
  merged-plan caches without changing label order.

#### Change Scope

- files:
  `native/pose_native_label_engine/src/hbm_inplace.cu`,
  `tests/parity/test_native_labeling.py`,
  `scripts/profile_scratch_breakdown.py`
- profiles benchmarked:
  direct native HBM microbenchmark, `single-h100-hbm-2mib`

#### Commands

```bash
scripts/build_native_label_engine.sh

(cd native/pose_native_label_engine && cargo test -q)

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/parity/test_native_labeling.py \
  tests/unit/test_native_hbm_microbench.py

PYTHONPATH=src .venv/bin/python -m pose.benchmarks.native_hbm_microbench \
  --label-count-m 65536 \
  --graph-parameter-n 15 \
  --hash-backend blake3-xof \
  --label-width-bits 256 \
  --device 0 \
  --repetitions 1 \
  --output-json .pose/benchmarks/native-hbm-microbench/20260323T135012Z-arithmetic.json

PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
import yaml
from pose.benchmarks.harness import run_benchmark
from pose.protocol.codec import load_json_file

plan = load_json_file(Path('.pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z/plan.json'))
profile = plan['profile']
with tempfile.TemporaryDirectory(prefix='pose-h100-hbm-2mib-arithmetic-cuda-rerun-') as temp_dir:
    profile_path = Path(temp_dir) / 'single-h100-hbm-2mib.yaml'
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding='utf-8')
    payload = run_benchmark(str(profile_path))
    summary = load_json_file(Path(payload['archive']['summary_path']))
    print(json.dumps({'run_directory': payload['archive']['run_directory'], 'summary': summary}, sort_keys=True))
PY

PYTHONPATH=src .venv/bin/python scripts/profile_scratch_breakdown.py \
  --mode cuda-hbm-in-place-arithmetic \
  --result-artifact .pose/benchmarks/single-h100-hbm-2mib/20260323T135136Z/run-001.result.json

PYTHONPATH=src .venv/bin/python -m pytest -q \
  --ignore tests/parity/test_reference_only_mode.py
```

#### Before Artifacts

- restored microbenchmark anchor:
  `.pose/benchmarks/native-hbm-microbench/20260323T131307Z-restored.json`
- current end-to-end HBM anchor:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T112146Z`

#### After Artifacts

- arithmetic HBM microbenchmark:
  `.pose/benchmarks/native-hbm-microbench/20260323T135012Z-arithmetic.json`
- accepted arithmetic HBM artifact:
  `.pose/benchmarks/single-h100-hbm-2mib/20260323T135136Z`

#### Metric Delta

Direct native HBM microbenchmark, compared against the restored anchor:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| micro `gpu_in_place` ms | 21,629.65 | 11,925.17 | -9,704.47 (-44.87%) |
| micro `scratch_peak_bytes` | 7,340,252 | 204 | -7,340,048 (-99.9972%) |
| micro `host_merged_plan_builds` | 15 | 0 | -15 (-100.00%) |
| micro `device_merged_plan_builds` | 15 | 0 | -15 (-100.00%) |
| micro `total_kernel_launches` | 655,016 | 655,016 | 0 (+0.00%) |

End-to-end `single-h100-hbm-2mib`, compared against the current anchor:

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms | 14,973 | 14,130 | -843 (-5.63%) |
| label_generation ms | 10,553 | 9,879 | -674 (-6.39%) |
| expected_response_prep ms | 3,684 | 3,563 | -121 (-3.28%) |
| graph_construction ms | 42 | 41 | -1 (-2.38%) |
| fast_phase ms | 205 | 205 | 0 (+0.00%) |
| verifier_cpu ms | 3,969 | 3,847 | -122 (-3.07%) |
| scratch_peak_bytes | 7,340,252 | 204 | -7,340,048 (-99.9972%) |

The new HBM scratch breakdown is exact:

- `pose_oracle_config = 204` bytes
- `host_merged_plan_cache = 0` bytes
- `device_merged_plan_cache = 0` bytes

#### Correctness Gates

- tests:
  `cargo test -q` passed for the native crate
- parity checks:
  the Python native/HBM parity slice passed (`9 passed`), including a new GPU
  profile check that asserts `scratch_peak_bytes < 1024` and both merged-plan
  build counters remain zero
- repository checks:
  `PYTHONPATH=src .venv/bin/python -m pytest -q --ignore tests/parity/test_reference_only_mode.py`
  passed (`164 passed`)
- hardware checks:
  both the direct microbenchmark and the accepted `single-h100-hbm-2mib`
  rerun returned `SUCCESS` with `declared_stage_copy_bytes=0`

#### Decision

- accepted because:
  the arithmetic recurrence removed the remaining CUDA HBM plan-cache scratch
  entirely, preserved the label schedule, and improved both the direct GPU fill
  benchmark and the stable end-to-end HBM benchmark
- conclusion:
  the HBM scratch figure was the mirrored merged-plan metadata, not an inherent
  requirement of the in-place device materialization path
- follow-up:
  if cooperative fused kernels are revisited later, they should keep using the
  arithmetic recurrence rather than reintroducing device index tables

## Template For Future Entries

Copy this section and increment the entry number.

### Entry 00N: Short Change Name

- date:
- git head:
- status: proposed | accepted | rejected | needs-follow-up
- hypothesis:

#### Change Scope

- files:
- profiles benchmarked:

#### Commands

```bash
# before

# after
```

#### Before Artifacts

- path:

#### After Artifacts

- path:

#### Metric Delta

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| total ms |  |  |  |
| label_generation ms |  |  |  |
| expected_response_prep ms |  |  |  |
| graph_construction ms |  |  |  |
| fast_phase ms |  |  |  |
| verifier_cpu ms |  |  |  |

#### Correctness Gates

- tests:
- parity checks:
- hardware checks:

#### Decision

- accepted because:
- rejected because:
- follow-up:
