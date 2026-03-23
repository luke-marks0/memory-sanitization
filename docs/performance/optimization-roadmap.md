# PoSE End-to-End Optimization Roadmap

## Purpose

This document defines how to make the PoSE-DB implementation faster without
weakening any correctness, claim-discipline, or spec-compliance guarantees.

The baseline conclusion from the current repository is simple:

- the fast phase is not the main bottleneck today;
- end-to-end runtime is dominated by full-graph label computation on both the
  prover and verifier sides;
- any optimization work must therefore focus first on graph reuse and label
  computation engines, not on transport micro-tuning.

## Non-Negotiable Guardrails

Every optimization proposal must satisfy all of the following:

1. It must preserve the exact graph semantics, challenge semantics, label
   semantics, attacker-budget accounting, cleanup behavior, and result schema.
2. It must not relax deadlines, reduce challenge count, or weaken calibration
   honesty in order to look faster.
3. It must not hide scratch buffers, stage copies, host tiers, GPU tiers, or
   other memory that the current claims are required to account for.
4. Any accelerated backend must remain behind parity gates against the current
   Python reference path.
5. Every change must record benchmark artifacts before and after the change in
   `docs/performance/optimization-log.md`.

Optimization is allowed to change implementation, but not meaning.

## Current Baseline

Environment used for the baseline in `optimization-log.md`:

- date: `2026-03-23`
- git head: `7f12e5fd8b41c8dca2f7b5cf8053c5db751a3fe3`
- python: `3.14.3`
- host: `Linux 6.5.0-45-generic`
- GPU: `NVIDIA H100 80GB HBM3`
- driver: `555.58.02`

Representative benchmark summaries:

| profile | total ms | label_generation ms | expected_response_prep ms | graph_construction ms | fast_phase ms | label+expected share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev-small` | 28995 | 13298 | 12264 | 1601 | 206 | 88.2% |
| `single-h100-hbm-small` | 29693 | 13778 | 12339 | 1489 | 207 | 88.0% |
| `single-h100-hybrid-small` | 30851 | 13592 | 13647 | 1673 | 205 | 88.3% |

Baseline artifacts:

- `.pose/benchmarks/dev-small/20260323T024202Z`
- `.pose/benchmarks/single-h100-hbm-small/20260323T024202Z`
- `.pose/benchmarks/single-h100-hybrid-small/20260323T024241Z`

The practical reading of these numbers is:

- prover `MaterializeLabels` is expensive;
- verifier `expected_response_prep` is equally expensive;
- `build_pose_db_graph(...)` is non-trivial but secondary;
- gRPC round execution is currently a small fraction of total runtime.

## Large-Profile Scaling Update

The accepted accelerated-labeling path was also exercised on larger HBM-only
development runs:

| profile | challenged bytes | estimated nodes | total ms | label_generation ms | expected_response_prep ms | graph_construction ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single-h100-hbm-1mib` | 1,048,576 | 7,012,350 | 71,414 | 27,596 | 13,493 | 14,703 |
| `single-h100-hbm-2mib` | 2,097,152 | 15,990,782 | 161,293 | 63,500 | 27,739 | 34,780 |

Artifacts:

- `.pose/benchmarks/single-h100-hbm-1mib/20260323T045538Z`
- `.pose/benchmarks/single-h100-hbm-2mib/20260323T045711Z`

Implications:

- label work is still the biggest runtime bucket, but graph construction is now
  about `20%` to `22%` of total and can no longer be treated as a minor cost
  on larger profiles;
- the current in-process graph cache is useful for repeated runs, but it does
  not solve cold one-shot large-profile runs because prover and verifier still
  each construct large Python graph objects;
- a true `1 GiB` challenged profile is not feasible in the current
  architecture:
  `m = 33,554,432`, `n = 24`, estimated graph nodes
  `= 20,266,876,926`.

## Hot Path Map

The present cold-path runtime is:

1. verifier builds the session plan and allocates leases;
2. verifier builds the graph locally;
3. prover builds the same graph during `SeedSession`;
4. prover materializes labels over the full graph into host and/or GPU regions;
5. verifier recomputes expected challenged labels with another full-graph label
   pass;
6. verifier performs 64 fast rounds and checks responses.

Current code locations:

- verifier run path: `src/pose/verifier/service.py`
- prover materialization path: `src/pose/prover/grpc_service.py`
- reference graph construction: `src/pose/graphs/construction.py`
- reference label computation: `src/pose/graphs/labeling.py`

## Evidence That Shapes Priority

One tempting idea is to compute only the challenged labels on the verifier.
That is not obviously a large win on the current graph family.

For the current 4096-label smoke profiles with 64 sampled challenges:

- total graph nodes: `557054`
- ancestor closure for sampled challenges: `549982`
- closure fraction: `98.7305%`

So a challenge-only recursive verifier path is unlikely to remove much work for
these graphs. It may still be useful for memory layout or native execution, but
it is not the first optimization target.

## Priority Order

### P0: Measurement First

Before invasive changes, improve the measurement story and keep the log strict.

Required discipline:

- record benchmark artifacts before every optimization patch;
- record benchmark artifacts after every optimization patch;
- report percent deltas for `total`, `label_generation`,
  `expected_response_prep`, `graph_construction`, and `fast_phase_total`;
- run the relevant correctness and parity tests before accepting the change.

Recommended follow-up instrumentation:

- split prover-side `SeedSession` graph build into its own timing field;
- split verifier-side control-plane time for `plan`, `lease`, `seed`,
  `materialize`, `prepare`, and `finalize`;
- add a timing field for session startup and process spawn separately from the
  protocol phases.

This is not a speedup by itself, but it is necessary to know what really sped
up.

### P1: Cache Immutable Graph Objects

The graph depends only on the graph descriptor, not on session seed.

Safe optimization:

- memoize `build_pose_db_graph(...)` by descriptor digest in-process on both
  prover and verifier;
- reuse the cached graph in verifier cold runs, calibration, and rechallenge;
- reuse the cached graph in prover `SeedSession`.

Why it is safe:

- the graph is immutable for a given descriptor;
- caching changes construction cost, not semantics;
- the digest already identifies the canonical graph shape.

Expected value:

- low-to-moderate;
- likely saves on the order of the current `graph_construction` cost on both
  sides.

Validation requirements:

- parity tests for descriptor digest and predecessor lists;
- benchmark log entries showing `graph_construction` reduction;
- no changes to result claims or calibration outputs.

### P1.5: Eliminate Cold Cross-Process Graph Rebuilds

The large HBM runs show that one-shot `graph_construction` is no longer a small
cost. The current in-process cache does not remove this for cold CLI runs
because prover and verifier still materialize large Python graph objects
independently.

Safe optimization direction:

- add a disk-backed or memory-mapped graph artifact keyed by descriptor digest;
- or replace full predecessor-tuple materialization with a compact or
  formula-driven graph representation that can generate predecessor lists on
  demand without building billions of Python objects;
- ensure prover and verifier consume the same canonical artifact or generator
  logic.

Why it is safe:

- the descriptor digest still names the canonical graph;
- only representation changes, not graph topology or challenge semantics;
- parity can still validate predecessor lists on sampled or exhaustive small
  cases.

Expected value:

- moderate on smoke profiles;
- high on larger one-shot runs where graph construction has already reached
  `14.7 s` at `1 MiB` and `34.8 s` at `2 MiB`.

Current status:

- accepted and implemented via a formula-driven implicit graph engine
- observed result on `single-h100-hbm-2mib`:
  `graph_construction = 34,780 ms -> 43 ms`
- important follow-on:
  some of that work moved into repeated Python predecessor regeneration inside
  `label_generation` and verifier `expected_response_prep`

### P1.6: Amortize Formula-Driven Predecessor Traversal

The formula-driven graph engine removed the cold build cost, but the same
predecessor rows are now regenerated multiple times per session in Python:

- prover successor-count prepass;
- prover materialization pass;
- verifier expected-response preparation.

Safe optimization direction:

- fuse successor-count and label-generation traversal where possible;
- keep the implicit graph, but emit predecessor rows through a lower-overhead
  iterator or compact temporary row buffer scoped to one session;
- avoid falling back to a persistent full predecessor table that recreates the
  original memory problem.

Why it is safe:

- it does not change graph topology, challenge ordering, or label bytes;
- it only changes how often the same canonical predecessor stream is regenerated
  inside one session;
- parity remains anchored to the preserved explicit reference builder.

Expected value:

- moderate-to-high;
- the first formula-driven benchmark still improved total runtime by `25.5%`,
  but `label_generation` rose by `26.1%` and
  `expected_response_prep` rose by `41.7%`, which is now the clearest next win.

Current status:

- accepted
- streaming predecessor specs improved verifier-heavy work on
  `single-h100-hbm-2mib`:
  `expected_response_prep = 39,295 ms -> 34,730 ms`
  and `verifier_cpu = 39,692 ms -> 35,132 ms`
- follow-on reusable hash payload templates then reduced prover-heavy work:
  `label_generation = 80,653 ms -> 68,154 ms`
  and `total = 116,078 ms -> 102,977 ms`
- remaining gap:
  prover `label_generation` is still the largest bucket at `68.2 s`, so the
  next bottleneck is the remaining per-label hash setup and prover bookkeeping
  inside `MaterializeLabels`

### P1.7: Reuse Preseeded Hash State Per Domain

The reusable payload-template work removed allocation and payload rebuilding,
but each label hash still reabsorbs the same source/internal domain prefixes on
every call.

Safe optimization direction:

- precompute base hasher states for source labels and for each internal
  predecessor-count domain;
- clone or copy those states per label, then update only the node index and
  predecessor-label bytes;
- use the same domain-separated byte stream as today so the digest remains
  identical.

Why it is safe:

- it changes only how the identical input bytes are fed into the hash backend;
- parity can compare the new path directly against the current reusable-payload
  implementation and the preserved reference helpers;
- no graph, challenge, memory-accounting, or cleanup semantics change.

Expected value:

- moderate;
- now that the obvious Python allocation has been removed, repeated hashing of
  static prefix material is the clearest remaining pure label-engine overhead
  on both prover and verifier paths.

Validation requirements:

- exact label parity for both `blake3-xof` and `shake256`;
- benchmark evidence on `single-h100-hbm-2mib` and at least one smoke profile;
- explicit rejection if the backend copy overhead outweighs the prefix-hash
  savings.

Current status:

- rejected for the current Python backend stack
- measured result on `single-h100-hbm-2mib`:
  `total = 102,977 ms -> 118,353 ms`
  and `label_generation = 68,154 ms -> 76,568 ms`
- practical reading:
  cloning backend hash objects per label is slower here than hashing the
  reusable fixed-layout payload directly
- next step:
  stop pursuing pure-Python hash-state cloning and move to either:
  prover bookkeeping reductions in `MaterializeLabels`, or
  a native accelerated label engine behind parity gates

### P2: Introduce an Accelerated Label Engine Behind Parity Gates

This is the highest-value path because label generation dominates both sides of
the runtime.

Safe optimization direction:

- keep the current Python path as the normative reference implementation;
- add an optional accelerated engine for full-graph label computation;
- use it in prover materialization and verifier expected-label preparation only
  after parity checks pass.

Candidate implementations:

- Rust extension for full-graph topological label generation;
- CUDA backend for GPU-targeted materialization where labels can be produced
  directly in HBM;
- mixed strategy: Rust on verifier, Rust or CUDA on prover depending on region
  type.

Why it is safe when done correctly:

- it preserves exact label bytes for the same graph and session seed;
- parity gates can compare full arrays or challenged labels against the Python
  reference path;
- correctness remains anchored in the existing reference semantics.

Validation requirements:

- exhaustive parity on small graphs;
- randomized parity on larger graphs;
- benchmark evidence on all smoke profiles before enabling by default;
- fallback to the Python path on any mismatch or unsupported host.

### P3: Remove Python Object Churn in Prover Materialization

The current materializer allocates many Python objects per node:

- `predecessor_labels` lists;
- `bytes(...)` conversions;
- `bytearray` scratch entries in dictionaries;
- repeated small reads and writes through attachment objects.

Safe optimization direction:

- replace per-node object churn with preallocated contiguous buffers;
- use `memoryview`-style slices where possible instead of fresh `bytes`
  objects;
- reduce scratch bookkeeping overhead with indexed storage instead of
  `dict[int, bytearray]` when the live set shape is predictable;
- avoid repeated `bytes(scratch_label)` copies if the hash backend can consume
  buffer views safely.

Why it is safe:

- it changes only storage mechanics, not the topological order or label bytes;
- scratch lifetime and zeroization can remain explicit;
- stage-copy accounting can remain unchanged.

Expected value:

- high on prover `label_generation`;
- possibly meaningful scratch reduction too.

Validation requirements:

- exact label parity against the current materializer;
- adversarial and cleanup tests unchanged;
- benchmark log entries showing `label_generation` improvement.

### P4: Reduce Verifier Recompute Cost Without Changing Semantics

Because the closure is almost the full graph, the likely verifier win is not
challenge-only recursion. The better targets are:

- use the same accelerated full-graph label engine as the prover;
- cache graph objects;
- reduce temporary allocations in `compute_node_labels(...)`.

Safe optimization direction:

- keep the same challenge schedule and expected bytes;
- optimize the engine, not the theorem;
- prefer shared backend logic for prover and verifier so parity work is not
  duplicated.

Expected value:

- high;
- currently `expected_response_prep` is about 42% to 44% of total runtime.
- still high at larger scale:
  `label_generation + expected_response_prep` is about `57%` of total at
  `2 MiB`.

### P5: Make Rechallenge the Preferred Fast-Iteration Benchmark

The cold path is expensive because it rematerializes and rederives everything.
That does not mean cold-path work is unimportant, but it does mean developer
iteration can be accelerated with a better rechallenge discipline.

Safe optimization direction:

- improve host-only rechallenge benchmarking and use it aggressively for
  fast-phase experiments;
- later extend the retained-session concept carefully to HBM and hybrid only if
  cleanup and claim semantics stay explicit.

Why it matters:

- it separates fast-phase transport experiments from full relabeling costs;
- it prevents false conclusions from cold-path noise.

### P6: Transport and RPC Optimization, But Only After the Above

The current fast phase is about `0.7%` of total runtime on smoke profiles.
That makes it a poor first target.

Possible later work:

- persistent channels across more control-plane RPCs;
- unary-to-streaming fast rounds with preserved one-round semantics;
- lower-overhead encoding for single-round messages;
- fewer Python allocations in `FastPhaseClient`.

Hard restriction:

- no batching that weakens the interpretation of a timed single-label round;
- no change that hides per-round deadline violations.

### P7: GPU-Heavy Optimizations for Large HBM Profiles

The smoke profiles are too small to expose meaningful `copy_to_hbm` cost. For
larger HBM sessions, the following become important:

- direct-in-HBM label materialization;
- pinned host staging where a staging path is unavoidable;
- async `cudaMemcpy` and `cudaMemset` with explicit synchronization points;
- reduced attach/detach overhead for CUDA IPC.

These should be pursued only after:

- the production HBM claim path exists;
- max-profile benchmarks are available;
- native parity gates are already in place.

## Optimizations That Are Explicitly Out of Bounds

The following do not qualify as acceptable performance work:

- changing graph family or graph descriptor rules;
- lowering `rounds_r` to reduce runtime;
- increasing deadline budgets to mask slower code;
- weakening zeroization or skipping cleanup verification;
- omitting uncovered host or GPU tiers from attacker-budget accounting;
- replacing the reference semantics with an accelerated path that lacks parity
  coverage;
- hiding stage buffers or making claim notes less honest.

## Recommended Execution Plan

Work order:

1. improve timing visibility and keep the optimization log strict;
2. reduce prover materializer object churn and predecessor-label load overhead;
3. optimize or accelerate the full-graph label engine further on the prover;
4. keep verifier recomputation on the compact implicit traversal path;
5. optimize verifier recomputation with the same backend where it is still hot;
6. only then spend time on transport and fast-phase micro-optimizations.

## How To Record Progress

Use `docs/performance/optimization-log.md` for every experiment.

For each entry:

- record the hypothesis;
- record the exact benchmark commands;
- record before artifacts;
- record after artifacts;
- record percent deltas;
- record correctness and parity gates that were run;
- state whether the change is accepted, rejected, or needs follow-up.
