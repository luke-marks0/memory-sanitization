# Repository Specification: Python PoSE Using Depth-Robust-Graph PoSE-DB

**Status:** Draft v2
**Audience:** Maintainers and contributors implementing the repository
**Normative language:** “must”, “must not”, “required”, “shall”, and “should” are normative unless a section is explicitly marked informative.

---

## 1. Purpose

This repository shall implement a proof of secure erasure (PoSE) over user-space accessible host memory and GPU HBM using the depth-robust-graph PoSE-DB protocol described in the bundled paper.

The repository is Python-first in orchestration and reference semantics, with optional native acceleration for performance-critical paths:

- Python owns the CLI, prover process, verifier process, protocol, region planning, memory allocators, benchmarking, result reporting, and the reference graph construction and labelling semantics.
- Optional Rust and CUDA components may accelerate graph construction, label generation, challenge handling, and HBM transfers, but they must remain semantically identical to the Python reference implementation.

The top priority is correctness and claim discipline:

1. The production protocol must be the graph-based PoSE-DB protocol grounded in the bundled paper’s formal model.
2. The prover must overwrite the challenged memory budget with the session’s graph-label state and then answer timed challenge rounds correctly.
3. The repository must keep prover and verifier responsibilities cleanly separated so that the protocol can be benchmarked, stressed, and iterated on without weakening the security story.
4. The repository must report theorem-level claims and engineering-level region claims honestly and separately.

---

## 2. Goals

### 2.1 Primary goals

The repository must:

- implement the graph-based PoSE-DB protocol using a depth-robust graph construction faithful to the bundled paper;
- implement a clean prover/verifier split as separate processes;
- implement PoSE sessions over:
  - host memory only,
  - GPU HBM only,
  - hybrid host + HBM memory layouts;
- support a CLI that runs the erasure protocol and outputs:
  - whether the session succeeded,
  - how much memory was covered,
  - what soundness model and bound were used,
  - how long each stage took;
- maximize coverage of user-space accessible memory on:
  - an initial single-H100 machine,
  - a later 8×H100 machine;
- include a comprehensive test suite that proves:
  - graph construction is correct,
  - label generation is deterministic and parity-checked,
  - the PoSE-DB protocol behaves correctly and fails safely,
  - timing calibration enforces the required `q < γ` condition for production profiles.

### 2.2 Secondary goals

The repository should:

- support local benchmarking and repeated rechallenge runs;
- make it easy to compare host-only, HBM-only, and hybrid profiles;
- support multiple hash backends behind one canonical session specification;
- provide a pure Python reference implementation and native accelerators with parity gates;
- preserve a strong audit trail from implementation choices back to the bundled paper.

---

## 3. Non-goals

The repository does not claim to solve or provide:

- kernel-level or firmware-level memory erasure;
- attestation of inaccessible, privileged, or kernel-owned memory;
- erasure guarantees for disk, SSD, NVMe, or remote storage;
- protection against a malicious OS kernel, malicious NVIDIA driver, or malicious device firmware;
- secure-hardware-based attestation;
- unconditional theorem-level guarantees for a concrete hash backend beyond the random-oracle heuristic used by the bundled paper;
- exclusive residency in a specific memory tier unless that stronger claim is separately documented and justified;
- production use of the paper’s unconditional O(n) communication PoSE protocol, except optionally as a benchmark or test comparator.

The repository’s security claims are limited to the declared protocol, the challenged user-space memory regions, the attacker-budget assumptions used by the verifier, and the trusted-computing-base assumptions documented below.

---

## 4. Normative protocol reference and core security statement

### 4.1 Bundled reference

The repository shall include the paper at:

  docs/references/software-based-memory-erasure-relaxed-isolation.pdf

For the purposes of this repository, this is the bundled reference paper.

The following are authoritative from the bundled paper unless this spec explicitly narrows or operationalizes them:

- the PoSE-DB session model;
- the graph-based PoSE scheme;
- the definitions of `(m, γ)`-depth-robustness, graph-restricted adversary, and in-place labelling;
- the single-round and multi-round soundness bounds.

The implementation target is the graph-based protocol corresponding to the paper’s formal model and depth-robust-graph construction, adapted to verifier-owned host and HBM leases.

### 4.2 Session success

A PoSE session is considered successful only if all of the following hold:

1. The verifier chose a session seed and graph/session parameters for the active profile.
2. The prover materialized the session label array
   `σ = l(o1) || l(o2) || ... || l(om)`
   into the leased challenged regions according to the canonical region plan.
3. The verifier executed `r` fast challenge-response rounds.
4. In every round:
   - the verifier challenged one uniformly random challenge-set position,
   - the prover returned the correct label,
   - the verifier received the response within the configured per-round deadline `Δ`.
5. The active profile’s calibration established a conservative `q` with `q < γ`.
6. The covered-byte fraction met the configured coverage threshold.
7. Cleanup and release semantics were carried out or any failure was explicitly reported.

### 4.3 Theorem-level security interpretation

Under the bundled paper’s model, a successful session means:

- the prover answered timed challenge rounds for labels derived from the session seed and active graph descriptor; and
- given the active attacker-budget assumption `M`, the verifier can report the corresponding soundness bound from the bundled paper.

For reporting, the verifier shall compute a bound of the form:

  Pr[success] <= (M' / m)^r + 2^(-w0)

where the active adversary model determines `M'` and `w0` as in the bundled paper:

- graph-restricted model:
  - `w0 = w_bits`
  - `M' = ceil(M_bits / w_bits)`

- general model:
  - `w0 = w_bits - log2(m) - log2(q)`
  - `M' = ceil(M_bits / w0)`

If implementation rounding is required, it must be conservative in the verifier’s favour.

### 4.4 Claim discipline

The repository shall distinguish two claim classes:

- formal claim:
  a theorem-based claim about local storage under the bundled paper’s model, using the configured `M`, `r`, `q`, `γ`, and `w_bits`;

- operational claim:
  an engineering claim that the canonical storage location for the challenged labels was a specific verifier-leased host/HBM region layout.

The repository must never present an operational claim as though it were fully implied by the theorem when additional local-memory-tier assumptions are required.

---

## 5. Memory allocation ownership

The verifier process owns the challenged memory allocations and leases them to the prover process:

- for host memory:
  shared anonymous mappings, `memfd`, or equivalent verifier-owned shared mappings;

- for HBM:
  verifier-owned CUDA allocations exported to the prover by CUDA IPC handles or an equivalent same-host device-sharing mechanism.

The verifier chooses the exact challenged user-space regions, can account for their size, can track their lifecycle, and can enforce cleanup semantics.

---

## 6. Threat model and assumptions

### 6.1 Attacker model

The prover process may be adversarial and may communicate with an external conspirator. It may attempt to:

- store only part of the label set and recompute missing labels on demand;
- keep the label state outside the challenged regions and answer from another local store;
- use unchallenged host memory, HBM, pinned buffers, or stage buffers as hidden state;
- partially overwrite a region rather than fully filling the planned label slots;
- replay stale session seeds or stale labels;
- exploit protocol ambiguity, race conditions, transport jitter, or stale manifests;
- answer after a delay that hides reconstruction, copying, or tier-crossing costs.

### 6.2 Distant-attacker assumption

The bundled paper replaces full isolation with a distant-attacker assumption.

This repository shall interpret that assumption operationally as follows:

- a remote conspirator may assist before and between fast rounds;
- during a single timed fast round, only the local prover state may be relied upon;
- the configured deadline `Δ` and the measured `q` must make it implausible for the prover to recompute missing labels fast enough within one round.

### 6.3 Random-oracle assumption

The bundled paper’s graph-based security proof is in the random-oracle model.

This repository shall therefore:

- state clearly that its theorem-based claims rely on the random-oracle heuristic for the selected concrete hash backend;
- treat the concrete hash backend as a practical instantiation of the paper’s `h`;
- avoid wording that implies a stronger theorem for the concrete backend than the paper provides.

### 6.4 Trusted computing base

The trusted computing base includes:

- the verifier process;
- the operating system kernel;
- the file and memory sharing primitives used for leases;
- the Python verifier implementation;
- the reference graph construction and label semantics;
- any enabled native acceleration modules;
- the hash backend used to instantiate the paper’s random oracle;
- the CUDA runtime and NVIDIA driver in HBM mode;
- the clock source and timing instrumentation used by the verifier.

### 6.5 Local-memory-budget assumption

The paper’s theorem reasons about the attacker’s local state budget `M` during the fast phase.

For this repository:

- all unchallenged local memory that could hold useful label state must be counted toward `M` unless a stronger exclusion mechanism is documented and justified;
- this includes, where applicable:
  - unchallenged host memory,
  - unchallenged HBM,
  - managed-memory mirrors,
  - pinned host buffers,
  - stage buffers not zeroized before the fast phase,
  - scratch files or mapped buffers usable at fast-round speed.

The verifier must report `M` honestly. It must not silently assume that an unchallenged local tier is unavailable to the attacker.

#### 6.5.1 Development-only lowered-`M` mode

For host-only development and test profiles, the repository may support a
development-only `process_budget_dev` mode in which the configured attacker
budget `M_dev` is set artificially below the machine's full local-memory
budget.

This mode exists only to make local testing, calibration, and regression runs
practical on ordinary rented or shared compute. It must not be used to justify
production or theorem-level claims about the prover's true full local memory.

When this mode is used:

- the active profile must be explicitly marked as development-only via
  `process_budget_dev`;
- the result artifact must state that `M` was lowered artificially for
  development;
- the verifier must record the configured development attacker budget;
- the prover launcher may apply best-effort process-local limits such as
  `RLIMIT_AS`, `RLIMIT_MEMLOCK`, and `RLIMIT_FSIZE`;
- the prover launcher may hide GPUs from the prover process;
- production profiles must not use this mode.

The process-local limit may be larger than the reported development attacker
budget because the Python prover runtime itself requires non-trivial address
space. This mode is therefore an engineering convenience, not a claim that the
process limit equals the prover's true full local-memory budget.

#### 6.5.2 What development-only lowering does not justify

The following do not justify reporting a lowered development `M` as though it
were the prover's true full local-memory budget:

- process-local `ulimit` or `RLIMIT_AS` limits;
- hiding GPUs from the process while leaving the surrounding host unchanged;
- a memory limit that applies only to one process but not to helper processes;
- advisory cleanup of temporary files or pinned buffers without hard limits.

Such mechanisms may still be useful for development and regression control, but
they must not be reported as production attacker-budget accounting.

### 6.6 Tier-specific claim caveat

The theorem-level claim is about local storage, not intrinsically about one physical memory tier.

Therefore:

- host-only, HBM-only, and hybrid sessions may all be benchmarked;
- a profile that challenges only one tier must include the other local tiers in the attacker-budget accounting unless they are explicitly constrained by additional documented mechanisms;
- the repository must not describe an HBM-only or host-only run as proving exclusive residency in that tier unless that stronger claim is separately justified.

### 6.7 Coverage claim

The repository may only claim coverage for:

- the challenged leased host-memory regions;
- the challenged leased HBM regions.

It must never report “box memory erased” or “full GPU erased” unless the reported coverage equals the measured user-space accessible capacity under the active profile and the attacker-budget accounting across remaining local tiers is made explicit.

---

## 7. Terminology

- `w_bits`:
  label width in bits, matching the bundled paper’s notation.
- `w_bytes`:
  `w_bits / 8`.
- `m`:
  number of challengeable labels in the active session.
- `γ`:
  minimum path-length parameter guaranteed by the active graph construction.
- `q`:
  conservative upper bound on hash-oracle queries or equivalent local recomputation work that can fit in one fast round under the active deadline.
- `M_bits`:
  assumed attacker local-state budget during the fast phase, in bits.
- `graph descriptor`:
  canonical description of the active graph family, parameters, challenge-set ordering, hash backend, and label width.
- `O(G)`:
  ordered challenge set of graph nodes used by the active session.
- `label array`:
  the ordered byte string `σ = l(o1) || ... || l(om)`.
- `slot`:
  one label-sized challengeable storage unit in a leased region.
- `region`:
  one challenged memory allocation, either host or HBM.
- `covered bytes`:
  bytes occupied by full challengeable slots.
- `slack bytes`:
  bytes in usable challenged regions that are not part of any full slot.
- `fast round`:
  one timed challenge-response exchange over a single challenge-set position.
- `lease`:
  a verifier-owned region handle granted to the prover.
- `graph-restricted`:
  used in the sense of the bundled paper.
- `rechallenge`:
  a new fast phase against an already resident label array without rebuilding it.
- `formal claim`:
  theorem-based claim under the paper’s model.
- `operational claim`:
  engineering claim about the intended physical placement of label slots.

---

## 8. High-level architecture

### 8.1 Components

The repository shall contain these top-level implementation components:

1. Bundled paper and proof notes
   - the reference PDF and implementation-facing notes.

2. Python reference graph and label library
   - deterministic graph construction,
   - challenge-set ordering,
   - reference hash encoding,
   - reference label derivation,
   - reference soundness calculations.

3. Optional native acceleration
   - Rust and/or CUDA modules for performance-critical paths,
   - always parity-gated against the Python reference.

4. Python prover
   - region intake,
   - session execution,
   - label generation,
   - materialization into host/HBM,
   - fast-round responses,
   - timing collection,
   - cleanup.

5. Python verifier
   - region leasing,
   - session planning,
   - seed generation,
   - expected-response preparation,
   - challenge generation,
   - per-round deadline enforcement,
   - soundness reporting,
   - final verdict construction.

6. CLI
   - user-facing entry points for prover, verifier, calibration, and benchmarking workflows.

7. Benchmark harness
   - named profiles,
   - repeatable calibration and timing runs,
   - artifact production.

### 8.2 Architectural principle: one protocol, two phases

The architecture must preserve one clean protocol with two phases:

- initialization phase:
  derive and store the graph-label state from the session seed;

- fast phase:
  issue timed single-label challenges against that resident state.

The repository must not reintroduce a separate multi-proof layering split. The graph-based PoSE-DB protocol itself is the required proof mechanism.

### 8.3 Reference-before-acceleration principle

The Python reference implementation is authoritative for semantics.

Native accelerators may optimize:

- implicit graph traversal,
- label generation,
- HBM writes,
- challenge lookup paths,
- hash backend throughput,

but they must not change:

- the graph family,
- the challenge distribution,
- the label equations,
- the challenge-set ordering,
- the acceptance rule,
- the reported soundness model.

---

## 9. Repository layout

The repository shall use the following structure or a structure equivalent in clarity and separation:

  repo/
    README.md
    LICENSE
    THIRD_PARTY_NOTICES.md
    pyproject.toml
    Makefile

    docs/
      repository-spec.md
      architecture.md
      threat-model.md
      protocol.md
      graph-construction.md
      security-model.md
      benchmarking.md
      result-schema.md
      references/
        software-based-memory-erasure-relaxed-isolation.pdf
      hardware/
        single-h100.md
        eight-h100.md

    proto/
      pose/v1/session.proto

    # optional native-acceleration directories when enabled
    rust/
      pose_reference_kernels/

    cuda/
      pose_label_kernels/
      pose_copy_kernels/

    src/pose/
      __init__.py
      version.py

      cli/
        main.py
        prover.py
        verifier.py
        bench.py
        calibrate.py

      common/
        errors.py
        timing.py
        units.py
        env.py
        logging.py
        cbor.py

      protocol/
        messages.py
        codec.py
        session_ids.py
        result_schema.py

      graphs/
        construction.py
        connectors.py
        factory.py
        ordering.py
        descriptors.py
        depth_robustness.py
        inplace.py
        implicit.py

      hashing/
        random_oracle.py
        encoding.py
        blake3_backend.py
        shake256_backend.py

      prover/
        service.py
        session.py
        planner.py
        labeler.py
        challenge.py
        cleanup.py
        regions.py
        memory/
          host.py
          gpu.py

      verifier/
        service.py
        policy.py
        deadlines.py
        leasing.py
        challenges.py
        expected_labels.py
        soundness.py
        result_writer.py

      benchmarks/
        harness.py
        profiles.py
        summarize.py
        calibration.py

    bench_profiles/
      dev-small.yaml
      single-h100-host-max.yaml
      single-h100-hbm-max.yaml
      single-h100-hybrid-max.yaml
      eight-h100-hbm-max.yaml
      eight-h100-hybrid-max.yaml

    scripts/
      gen_graph_vectors.py
      run_lab_matrix.sh
      profile_hash_q.py

    tests/
      unit/
      parity/
      integration/
      e2e/
      adversarial/
      hardware/
      performance/

### 9.1 Separation invariant

The verifier package must not import prover-only modules.
The prover package must not contain verifier decision logic.
Shared data structures must live only in `src/pose/protocol/` and `src/pose/common/`.
Reference graph semantics must live only in `src/pose/graphs/` and `src/pose/hashing/`.

---

## 10. Protocol reference and conformance policy

### 10.1 Required production protocol

The required production protocol is the graph-based PoSE-DB protocol from the bundled paper.

The repository may include the unconditional protocol from the same paper only as:

- a test oracle,
- a research comparator,
- or a benchmark baseline.

It must not be the default production mode described by this spec.

### 10.2 Paper sections with implementation force

The implementation shall be traceable to:

- the paper’s formal PoSE-DB model for session structure and attacker model;
- the paper’s graph-based PoSE scheme for `Setup`, `Precmp`, `Chal`, `Resp`, and `Vrfy`;
- the paper’s depth-robust graph construction for the active graph family and its arbitrary-`m` extension.

### 10.3 Conformance boundary

Optimizations are allowed only if they preserve the semantics of:

- graph construction,
- node ordering,
- predecessor ordering,
- challenge-set ordering,
- label derivation,
- challenge distribution,
- response verification,
- soundness calculation.

Any change outside that boundary is a protocol change and requires a spec revision.

### 10.4 Optimization boundary

The repository may optimize:

- graph generation by implicit representation,
- recursion scheduling,
- host/HBM materialization strategy,
- hash throughput,
- verifier precomputation strategy,
- transport framing for the fast phase,

but it must not change the acceptance condition or what bytes are considered covered.

### 10.5 Production shortcut ban

Production code must not:

- replace the active graph family with a toy or unproven substitute;
- answer fast rounds from an undeclared external response table rather than the resident session label state;
- compress the label array with an undeclared scheme and expand it on the fast path while still claiming the covered bytes as resident;
- silently fall back to an untimed or post-hoc verification mode;
- change the fast-round challenge distribution away from independent uniform sampling over the global challenge set without separate analysis;
- use unified/managed GPU memory in production mode while reporting HBM-specific coverage without making that downgrade explicit;
- keep stage copies outside the challenged regions past the fast-phase start without counting them toward the attacker budget.

CI shall include checks that fail if production builds enable banned testing or shortcut flags.

---

## 11. Reference implementation and acceleration policy

### 11.1 Python reference scope

Python must own, from day one:

- the reference graph factory;
- the canonical graph descriptor encoding;
- the canonical challenge-set ordering;
- the canonical label derivation logic;
- the soundness calculator;
- the planner that maps label slots into leased regions;
- the verifier logic.

### 11.2 Native acceleration policy

Rust and CUDA modules may accelerate:

- implicit graph traversal,
- connector evaluation,
- label generation,
- HBM copy and lookup paths,
- hash backends.

They must not become authoritative for semantics until parity tests pass.

### 11.3 Promotion policy

A native accelerated path may replace the Python reference path for production execution only when the specific component:

1. has a clear semantic boundary;
2. has exhaustive small-case parity vectors;
3. passes deterministic equivalence tests;
4. does not weaken the soundness report or claim scope.

### 11.4 Implicit-graph requirement

Because the paper’s graph family has superlinear node count in `m`, production implementations shall represent the graph implicitly.

They must not fully materialize all vertices and edges for large production sessions.

Explicit graph materialization is allowed only for:

- small reference tests,
- exhaustive small-`n` proofs,
- debugging tools.

---

## 12. Graph construction and label semantics

### 12.1 Required graph family

The required production graph family is:

  pose-db-drg-v1

This denotes a faithful implementation of the bundled paper’s depth-robust graph construction and its arbitrary-`m` extension.

### 12.2 Arbitrary-`m` construction

For a target label count `m`, the graph factory shall choose the smallest integer `n` such that:

  2^(n+1) >= m

and construct a graph `G` with an ordered challenge set `O(G)` such that:

- `|O(G)| = m`
- the graph is `(m, 2^n)`-depth-robust with respect to `O(G)`
- the graph can be labelled in-place with respect to `O(G)`

The resulting `γ` for the session is therefore:

  γ = 2^n

### 12.3 Node IDs and ordering

The implementation must define:

- a deterministic topological node numbering;
- a deterministic predecessor ordering for every node;
- a deterministic ordered challenge set `O(G) = (o1, ..., om)`.

These rules must be versioned as part of the graph descriptor.

### 12.4 Label width

`w_bits` must be:

- byte aligned,
- at least 128 bits,
- 256 bits by default in production profiles.

`w_bytes` is derived from `w_bits`.

### 12.5 Session seed and random-oracle instantiation

The verifier shall choose a fresh session seed of at least 256 bits for every new session.

The implementation shall instantiate the paper’s hash oracle as a concrete, domain-separated backend over:

- the session seed,
- the graph descriptor digest,
- the encoded node/predecessor-label input.

The default backend shall be:

  blake3-xof

The repository may also provide:

  shake256

or other backends, provided that:

- the backend is declared in the session plan,
- both prover and verifier use the same backend,
- parity tests exist where required.

### 12.6 Graph descriptor digest

The graph descriptor shall be encoded in deterministic CBOR and hashed to produce:

  graph_descriptor_digest

This digest must bind:

- graph family ID,
- `m`,
- `n`,
- `γ`,
- node-ordering version,
- challenge-set ordering version,
- hash backend,
- `w_bits`.

### 12.7 Label equation

Let the active graph be `G` and the ordered predecessor list of node `v` be `(v1, ..., vd)`.

The canonical label semantics are:

- if `v` has no predecessors:
  `l(v) = h_session(tag_input || encode(v))`

- otherwise:
  `l(v) = h_session(tag_internal || encode(v) || encode(d) || l(v1) || ... || l(vd))`

The exact byte encoding of `encode(v)` and `encode(d)` must be fixed, documented, and parity-tested.

### 12.8 Challenge-set indexing

The fast phase shall challenge a position `i` in the ordered challenge set, not an ad hoc byte offset.

The response for challenge position `i` is the label `l(oi)`.

This indexing must map deterministically to one physical slot in one leased region.

---

## 13. Challenged memory layout

### 13.1 Label array

The session label array is:

  σ = l(o1) || l(o2) || ... || l(om)

This is the exact byte sequence that the session claims as covered state.

### 13.2 Region slot mapping

For each challenged region:

- the verifier computes `slot_count = floor(usable_region_bytes / w_bytes)`;
- only full slots count toward covered bytes;
- no slot may straddle a region boundary.

The global session label count is:

  m = sum(slot_count over all challenged regions)

### 13.3 Global ordering

The verifier shall define a canonical global slot order by concatenating the regions in deterministic plan order and, within each region, enumerating slots by increasing offset.

This global slot order must correspond one-to-one with the ordered challenge set `O(G)`.

### 13.4 Slack bytes

Bytes in a challenged region that do not fit a full slot are slack bytes.

Slack bytes:

- must not be counted as covered bytes;
- must be reported separately;
- should be zeroized on cleanup;
- may be deterministically initialized for hygiene, but they are not part of the theorem-level proof.

### 13.5 Coverage metrics

The repository shall report:

- `covered_bytes`
- `slack_bytes`
- `coverage_fraction = covered_bytes / usable_challenged_bytes`

A session may not report success unless the profile’s coverage threshold is met.

Default threshold:

  0.99

The expected value for large profiles should be very close to 1.0 because slack is bounded by less than one slot per region.

---

## 14. In-place labelling requirements

### 14.1 Algorithmic requirement

The prover’s generation strategy must be faithful to the paper’s in-place property:

- aside from the destination label slots and bounded scratch buffers,
  it must not require a second `O(m)`-sized auxiliary label store.

### 14.2 What counts as extra state

The repository shall distinguish:

- covered state:
  the session label array occupying challengeable slots;

- algorithmic scratch:
  working memory needed to generate the label array;

- platform overhead:
  runtime, allocator, driver, and transport buffers.

Only covered state counts toward covered bytes.

Scratch and platform overhead must be measured and reported separately where practical.

### 14.3 Verifier resource asymmetry

The verifier is allowed to use more memory and time than the prover.

In particular, the verifier may:

- pre-sample the challenge schedule;
- precompute all expected responses for that schedule;
- cache expected responses outside the timed path.

The in-place requirement applies to the prover’s session-state generation path, not to the verifier.

### 14.4 Staging rules

A prover may stage labels outside the final challenged regions during initialization only if one of the following happens before the fast phase starts:

- the stage copy is zeroized and released; or
- the stage copy is counted into the attacker-budget accounting and reflected in claim notes.

Production profiles must not hide surviving stage copies.

### 14.5 HBM materialization rule

If a profile claims HBM coverage, the canonical covered slots for that part of the session must be HBM-resident leased slots.

Host shadows used during initialization must follow the staging rules above.

---

## 15. Deadline calibration and soundness policy

### 15.1 Required profile inputs

Every production profile must define, at minimum:

- `w_bits`
- graph family ID
- hash backend
- challenge rounds `r` or a target soundness bound from which `r` is derived
- per-round deadline `Δ`
- adversary model:
  `general` or `graph_restricted`
- attacker-budget assumption `M_bits` or `M_bytes`
- coverage threshold
- transport mode for the fast phase

Profiles may additionally define a prover sandbox policy for development-only
execution. If present, that policy must specify:

- that the mode is `process_budget_dev`;
- any process-local cap used for the prover child;
- whether GPU visibility must be suppressed; and
- any additional `prlimit`-style hardening such as `memlock` or file-size
  limits.

Production profiles must not rely on such a policy to define theorem-level `M`.

### 15.2 Calibrated `q`

The verifier shall derive a conservative `q` for the active profile.

`q` must upper-bound the amount of local recomputation work available inside one fast round under the chosen deadline.

At minimum, the calibration must include:

- fastest measured local hash-evaluation throughput on the active backend;
- transport and serialization overhead;
- resident lookup latency for a stored label;
- a safety margin.

If a conservative `q` cannot be derived, the profile is invalid.

### 15.3 Required inequality

A production profile is invalid unless:

  q < γ

This must be checked and recorded in the result artifact.

### 15.4 Round selection

If `r` is not fixed directly by the profile, the verifier shall choose the smallest `r` such that the reported soundness bound is at most the profile’s target bound.

If the chosen adversary model yields:

- `w0 <= 0`, or
- `M' / m >= 1`, or
- an additive `2^(-w0)` term that already exceeds the target bound,

then the profile must fail validation rather than report a misleading success probability.

### 15.5 Challenge distribution

Each fast round must challenge one independently sampled position uniformly from `[0, m)`.

Sampling is with replacement.

Repeated challenges are allowed.

Per-region quota schedules may be used only in diagnostic modes unless separately analysed and explicitly marked as non-theorem-preserving.

### 15.6 Calibration artifacts

The benchmark harness shall persist calibration evidence for each production profile, including:

- resident lookup latency distribution;
- fastest local hash throughput;
- derived `q`;
- `γ`;
- `q/γ` margin;
- any measured alternate-store copy timings used to justify operational region claims.

---

## 16. Memory model

### 16.1 Common requirements

The repository shall operate only on user-space accessible memory.

All accounting must distinguish:

- total detected capacity,
- reserved bytes,
- usable challenged bytes,
- covered bytes,
- slack bytes,
- declared attacker-budget bytes.

### 16.2 Host memory backend

The host backend shall support:

- anonymous shared mappings or `memfd`;
- optional `mlock`;
- optional huge pages;
- NUMA affinity controls where practical;
- explicit zeroization on teardown.

The verifier creates the host region and passes a lease handle to the prover.

### 16.3 GPU HBM backend

The HBM backend shall support:

- device memory allocations only;
- no unified/managed memory in production mode;
- CUDA IPC handle transfer;
- explicit zeroization on teardown where supported;
- per-device accounting.

A profile that claims HBM coverage must account for any host-side shadow copies under the staging rules.

### 16.4 Region planner

The planner shall maximize fill subject to reserve policy.

Inputs:

- available host bytes;
- per-GPU available HBM bytes;
- guard/reserve bytes;
- `w_bits`;
- graph family;
- target fill ratio;
- adversary model;
- attacker budget;
- supported backends.

Outputs:

- region plan;
- per-region slot counts;
- global `m`;
- graph parameter `n`;
- `γ`;
- expected slack;
- expected coverage fraction;
- claim notes if unchallenged tiers materially affect `M`.

### 16.5 Attacker-budget accounting across tiers

The planner must account for all local tiers that remain usable to the attacker during the fast phase.

For example:

- a host-only session on a machine with large free HBM must either:
  - include that HBM in `M`, or
  - explicitly constrain it by a separate documented mechanism;

- an HBM-only session on a machine with large free DRAM must either:
  - include that DRAM in `M`, or
  - explicitly constrain it by a separate documented mechanism.

This accounting is mandatory for theorem-level honesty.

---

## 17. Prover/verifier protocol

### 17.1 Transport

The repository shall separate:

- control plane:
  default gRPC over Unix domain sockets, or loopback TCP if Unix sockets are unavailable;

- fast phase:
  a low-overhead timed challenge channel.

The fast phase may use gRPC only if the profile’s calibration includes the resulting jitter and still establishes a valid `q < γ`.

A lower-overhead IPC transport is preferred for production fast rounds.

### 17.2 Normative message flow

The normative message flow is:

1. `Discover`
2. `PlanSession`
3. `LeaseRegions`
4. `SeedSession`
5. `MaterializeLabels`
6. `PrepareFastPhase`
7. `RunFastPhase`
8. `Finalize`
9. `Cleanup`

### 17.3 PlanSession

The verifier creates a session plan containing at minimum:

- session ID;
- session seed;
- graph family ID;
- graph parameter `n`;
- `m`;
- `γ`;
- `w_bits`;
- graph descriptor digest;
- hash backend;
- region plan;
- deadline policy;
- derived `q`;
- challenge-round count `r`;
- adversary model;
- attacker-budget assumption;
- cleanup policy.

### 17.4 LeaseRegions

The verifier sends lease handles for each challenged region.

Each lease must include:

- region ID;
- region type (`host` or `gpu`);
- usable bytes;
- slot count;
- slack bytes;
- lease handle;
- lease expiry;
- cleanup policy.

### 17.5 SeedSession

The verifier sends the session seed and graph/session parameters to the prover.

The prover must treat these as the only inputs for the covered label state, aside from the deterministic graph and hash semantics.

### 17.6 MaterializeLabels

The prover:

- constructs the active graph implicitly;
- computes the labels for the ordered challenge set `O(G)`;
- writes each label into its assigned slot in the leased regions;
- clears any stage copies or declares them for attacker-budget accounting;
- returns generation metadata and timings.

The metadata must include:

- graph descriptor digest;
- per-region covered bytes;
- per-region slack bytes;
- any declared stage-copy bytes remaining before fast phase.

### 17.7 PrepareFastPhase

Before the fast phase starts:

- the verifier may sample the entire challenge schedule;
- the verifier may precompute all expected response labels for that schedule;
- the verifier must keep undisclosed future challenges hidden from the prover;
- the prover must not retain undeclared stage copies.

### 17.8 RunFastPhase

For each of `r` rounds:

- the verifier chooses or reveals one challenge index `i`;
- the verifier starts its round timer immediately before challenge emission;
- the prover returns the bytes in slot `i`;
- the verifier records arrival time and verifies the returned label;
- the round fails if the response is wrong or late.

The timed path must include challenge transmission, prover lookup, any necessary host/HBM read, and response transmission.

A production acceptance verdict requires all rounds to succeed.

The verifier may continue after a failure only in diagnostic mode; the verdict remains failure.

### 17.9 Finalize

The verifier computes:

- the final verdict;
- the soundness bound for the declared adversary model;
- covered-byte and slack-byte metrics;
- any claim notes about unchallenged local tiers or declared stage copies.

It then writes the canonical result artifact.

### 17.10 Cleanup

Cleanup shall:

- zero challenged regions according to policy;
- zero slack bytes where practical;
- release leases and handles;
- record cleanup status in the result artifact.

Cleanup failure must be reported even if the fast phase succeeded.

---

## 18. Session result and verdicts

### 18.1 Allowed verdicts

A session verdict must be one of:

- `SUCCESS`
- `WRONG_RESPONSE`
- `DEADLINE_MISS`
- `CALIBRATION_INVALID`
- `COVERAGE_BELOW_THRESHOLD`
- `RESOURCE_FAILURE`
- `CLEANUP_FAILURE`
- `PROTOCOL_ERROR`

### 18.2 Required output fields

Every verifier run must emit, at minimum:

- `success`
- `verdict`
- `session_id`
- `profile_name`
- `graph_family`
- `graph_parameter_n`
- `graph_descriptor_digest`
- `label_width_bits`
- `label_count_m`
- `gamma`
- `hash_backend`
- `session_seed_hex` or a reproducible commitment to it
- `adversary_model`
- `attacker_budget_bytes_assumed`
- `target_success_bound`
- `reported_success_bound`
- `soundness_model`
- `deadline_us`
- `q_bound`
- `rounds_r`
- `accepted_rounds`
- `host_total_bytes`
- `host_usable_bytes`
- `host_covered_bytes`
- `gpu_devices`
- `gpu_usable_bytes_by_device`
- `gpu_covered_bytes_by_device`
- `covered_bytes`
- `slack_bytes`
- `coverage_fraction`
- `max_round_trip_us`
- `cleanup_status`
- `claim_notes`
- `timings_us` or `timings_ms`
- `environment`

If a development-only prover sandbox policy affected `M`, the result artifact
must make that explicit in `claim_notes` or equivalent structured fields,
including at minimum:

- sandbox mode identifier;
- statement that the attacker budget was lowered for development only;
- configured development attacker-budget value;
- configured process-local memory cap, if any; and
- whether GPUs were explicitly hidden from the prover.

### 18.3 Timing breakdown

The timing breakdown must include at least:

- `discover`
- `region_leasing`
- `allocation`
- `graph_construction`
- `challenge_schedule_prep`
- `expected_response_prep`
- `label_generation`
- `copy_to_host`
- `copy_to_hbm`
- `stage_buffer_cleanup`
- `fast_phase_total`
- `verifier_check_total`
- `cleanup`
- `total`

The fast-phase report should also include at least:

- `round_trip_p50`
- `round_trip_p95`
- `round_trip_p99`
- `round_trip_max`

### 18.4 Result artifact example

{
  "success": true,
  "verdict": "SUCCESS",
  "session_id": "2026-03-20T12:00:00Z-8e2d7d2a",
  "profile_name": "dev-small",
  "graph_family": "pose-db-drg-v1",
  "graph_parameter_n": 20,
  "graph_descriptor_digest": "sha256:6a8c4d7d2d6e9c4b...",
  "label_width_bits": 256,
  "label_count_m": 4194304,
  "gamma": 1048576,
  "hash_backend": "blake3-xof",
  "session_seed_hex": "9b0d7a7f...",
  "adversary_model": "general",
  "attacker_budget_bytes_assumed": 33554432,
  "target_success_bound": 1e-9,
  "reported_success_bound": 6.1e-11,
  "soundness_model": "random-oracle + distant-attacker + calibrated q<gamma",
  "deadline_us": 2500,
  "q_bound": 4096,
  "rounds_r": 128,
  "accepted_rounds": 128,
  "host_total_bytes": 17179869184,
  "host_usable_bytes": 1073741824,
  "host_covered_bytes": 1073741824,
  "gpu_devices": [],
  "gpu_usable_bytes_by_device": {},
  "gpu_covered_bytes_by_device": {},
  "covered_bytes": 1073741824,
  "slack_bytes": 0,
  "coverage_fraction": 1.0,
  "max_round_trip_us": 611,
  "cleanup_status": "ZEROIZED_AND_RELEASED",
  "claim_notes": [
    "formal claim is about local storage under the configured attacker budget",
    "no surviving stage buffers declared before fast phase"
  ],
  "timings_ms": {
    "discover": 3,
    "region_leasing": 11,
    "allocation": 16,
    "graph_construction": 28,
    "challenge_schedule_prep": 2,
    "expected_response_prep": 104,
    "label_generation": 863,
    "copy_to_host": 0,
    "copy_to_hbm": 0,
    "stage_buffer_cleanup": 2,
    "fast_phase_total": 74,
    "verifier_check_total": 8,
    "cleanup": 12,
    "total": 1123
  },
  "environment": {
    "python_version": "3.x",
    "native_accel": ["rust"],
    "cuda_runtime": null,
    "driver_version": null
  }
}

---

## 19. CLI requirements

### 19.1 Commands

The CLI shall expose the following top-level personas.

Prover:
  pose prover serve --config prover.toml
  pose prover inspect
  pose prover self-test

Verifier:
  pose verifier run --profile single-h100-hybrid-max
  pose verifier run --plan plan.yaml --json
  pose verifier rechallenge --session-id <id>
  pose verifier verify-record result.json
  pose verifier calibrate --profile single-h100-hbm-max

Benchmark:
  pose bench run --profile single-h100-hbm-max
  pose bench matrix --profiles bench_profiles/
  pose bench summarize results/*.json

Optional developer utility:
  pose graph inspect --m 1048576 --json

### 19.2 Exit codes

The CLI must use stable exit codes:

- `0` for success;
- nonzero for all failures.

The human-readable summary must never be the only output path; JSON must always be available.

### 19.3 Human-readable summary

The human summary must include:

- session verdict;
- covered bytes and slack bytes by tier;
- `m`, `γ`, `r`, `Δ`, and `q`;
- attacker-budget assumption;
- adversary model;
- reported success bound;
- total runtime;
- fast-phase timing summary;
- cleanup status.

It must not overstate the claim scope.

---

## 20. Test suite

The test suite is mandatory and central to repository credibility.

### 20.1 Test categories

The repository shall contain:

- `unit`
- `parity`
- `integration`
- `e2e`
- `adversarial`
- `hardware`
- `performance`

### 20.2 Paper-conformance tests

These tests prove that the repository still implements the intended protocol.

Required checks:

- the bundled paper exists at the documented path;
- the graph family ID and arbitrary-`m` factory produce the expected `m` and `γ`;
- the challenge distribution is uniform over the global challenge set;
- the soundness calculator matches the equations used by the bundled paper;
- production code does not enable banned shortcut paths.

### 20.3 Parity tests: Python reference vs native accelerators

Parity tests must compare native outputs against the Python reference for:

- graph descriptor encoding;
- node ordering;
- predecessor ordering;
- challenge-set ordering;
- label derivation for fixed seeds and backends;
- region slot mapping;
- expected response generation.

The expected result is bit-for-bit equality.

### 20.4 Graph property tests

Required cases:

- exhaustive depth-robustness checks for small graphs;
- exhaustive arbitrary-`m` construction checks for small `m`;
- small-`n` in-place-labelling checks;
- large-case structural invariant checks for implicit graph construction.

### 20.5 Label-correctness tests

Required cases:

- correct seed and graph descriptor produce accepted labels;
- wrong seed fails;
- wrong hash backend fails;
- wrong predecessor ordering fails;
- wrong node-ordering version fails;
- stale session seed replay fails.

### 20.6 Protocol tests

Required cases:

- version mismatch handling;
- malformed message handling;
- timeout handling;
- duplicate challenge handling;
- repeated session ID rejection;
- fast-phase transport fallback reporting;
- verifier/prover restart recovery where applicable.

### 20.7 In-place and staging tests

Required cases:

- prover generation path does not allocate a second `O(m)` label store;
- stage buffers are cleared before the fast phase or correctly declared;
- HBM materialization paths do not silently leave host shadows in production mode.

### 20.8 Adversarial tests

The adversarial suite must explicitly test:

- partial label storage with on-demand recomputation;
- stale label replay under a new seed;
- challenge answers served from the wrong slot;
- hidden host shadow during HBM profile;
- hidden HBM shadow during host profile;
- sparse writes to a challenged region;
- timeout caused by post-challenge recomputation;
- timeout caused by post-challenge copy-in;
- incorrect `q` calibration causing profile invalidation;
- mismatch between declared and actual attacker-budget accounting.

### 20.9 Hardware tests

Self-hosted or lab hardware jobs must include:

- single-H100 host-only;
- single-H100 HBM-only;
- single-H100 hybrid;
- 8×H100 HBM-only;
- 8×H100 hybrid.

Every hardware job must archive:

- result JSON;
- benchmark logs;
- calibration artifacts;
- environment snapshot;
- native-acceleration versions;
- CUDA runtime and driver versions;
- GPU inventory.

### 20.10 Performance regression tests

Nightly performance tests must detect regressions in:

- label-generation time;
- HBM transfer time;
- fast-phase latency;
- `q` margin;
- coverage fraction;
- total runtime.

---

## 21. Benchmarking

### 21.1 Benchmark classes

The benchmark harness shall support:

- `cold`
- `warm`
- `rechallenge`

Definitions:

- `cold`:
  includes first-run setup costs and calibration-relevant initialization;

- `warm`:
  repeated full runs with steady-state caches and warmed runtime;

- `rechallenge`:
  no relabelling or rematerialization, only new fast rounds against resident labels.

### 21.2 Required benchmark profiles

The repository must include these named profiles:

- `dev-small`
- `single-h100-host-max`
- `single-h100-hbm-max`
- `single-h100-hybrid-max`
- `eight-h100-hbm-max`
- `eight-h100-hybrid-max`

### 21.3 Benchmark profile fields

Each profile must define, at minimum:

- target devices;
- reserve policy;
- host target fraction;
- per-GPU target fraction;
- `w_bits`;
- graph family;
- hash backend;
- adversary model;
- attacker-budget assumption;
- target success bound or fixed `r`;
- deadline policy;
- `q` calibration policy;
- cleanup policy;
- repetition count.

### 21.4 Reported benchmark metrics

Every benchmark summary must report:

- success rate;
- mean / p50 / p95 / p99 timings;
- deadline miss rate;
- coverage fraction;
- slack bytes;
- `q` and `γ`;
- attacker-budget assumption;
- reported success bound;
- per-device HBM coverage;
- verifier CPU time;
- rechallenge performance.

---

## 22. Build and packaging

### 22.1 Python packaging

Use a standard Python package layout with pinned dependencies.

### 22.2 Native packaging

If native acceleration is added, use a packaging arrangement that makes it reproducible.

### 22.3 Python/native integration

Any future Python/native integration must have a reproducible build path and the
same parity and CI requirements as the reference implementation.

CUDA modules may be built separately but must integrate into the same parity and CI story.

### 22.4 Reproducibility

The repository must have:

- pinned Python dependencies;
- a Cargo lockfile if Rust is used;
- reproducible local build instructions;
- a single-command test path;
- a single-command calibration path;
- a single-command benchmark path.

### 22.5 Required developer commands

At minimum:

  make build
  make test
  make test-parity
  make test-graphs
  make test-hardware
  make calibrate PROFILE=single-h100-hbm-max
  make bench PROFILE=single-h100-hbm-max

---

## 23. Documentation requirements

The repository must ship the following documents:

- `docs/repository-spec.md`
- `docs/architecture.md`
- `docs/threat-model.md`
- `docs/protocol.md`
- `docs/graph-construction.md`
- `docs/security-model.md`
- `docs/benchmarking.md`
- `docs/result-schema.md`
- `docs/references/software-based-memory-erasure-relaxed-isolation.pdf`

Each document must be consistent with this spec.

If an engineering document disagrees with this file, this file wins.

If a theorem statement or formal definition is quoted or paraphrased here and there is a conflict with the bundled paper, the bundled paper wins unless this spec explicitly states that it is imposing a narrower implementation requirement.

---

## 24. Milestones

### 24.1 Phase 0 — foundation

Deliverables:

- bundled paper in the repository;
- Python reference graph factory;
- Python reference labeler;
- session and result schemas;
- basic prover/verifier skeleton;
- parity harness scaffolding.

Exit criteria:

- small reference sessions run end-to-end;
- graph descriptor and label vectors are deterministic;
- soundness calculator matches the chosen formulas.

### 24.2 Phase 1 — host-memory PoSE-DB

Deliverables:

- verifier-leased host memory regions;
- canonical slot mapping and label-array layout;
- host-only fast phase;
- verifier result artifact;
- host-only CLI flow.

Exit criteria:

- successful host-only PoSE-DB on development hardware;
- adversarial host-memory tests passing;
- calibrated profile with valid `q < γ`.

### 24.3 Phase 2 — single-H100 HBM PoSE-DB

Deliverables:

- HBM leasing via CUDA IPC;
- HBM materialization path;
- HBM fast phase;
- single-H100 HBM benchmark profiles.

Exit criteria:

- successful HBM session on one H100;
- HBM calibration artifacts archived;
- any host-shadow caveats reported honestly.

### 24.4 Phase 3 — hybrid host + HBM

Deliverables:

- mixed-region sessions;
- unified result reporting;
- planner across host + HBM;
- hybrid benchmark profiles.

Exit criteria:

- successful hybrid session on one H100 box;
- attacker-budget accounting across both tiers validated.

### 24.5 Phase 4 — 8×H100 scale-out

Deliverables:

- per-device worker model;
- region aggregation across 8 GPUs;
- multi-GPU benchmark harness;
- lab automation.

Exit criteria:

- successful 8×H100 HBM-only and hybrid sessions;
- archived calibration and result artifacts.

### 24.6 Phase 5 — accelerator promotion

Deliverables:

- native host and GPU labelers;
- parity closure against the Python reference;
- formal promotion process for selected accelerated components.

Exit criteria:

- promoted components have parity reports;
- no regression in reported claim scope or soundness accounting.

---

## 25. Acceptance criteria

The repository shall not be considered complete until all of the following are true:

1. A verifier can run a successful end-to-end host-memory PoSE-DB session.
2. A verifier can run a successful end-to-end HBM session on a single H100.
3. The same protocol scales to an 8×H100 box.
4. Every successful session uses the bundled paper’s graph-based protocol or a formally equivalent implementation allowed by this spec.
5. Production profiles demonstrate and record a valid `q < γ`.
6. Native accelerators pass parity tests against the Python reference.
7. CLI output includes verdict, coverage, attacker-budget assumptions, soundness model, and timings.
8. Benchmark profiles exist and produce machine-readable artifacts.
9. Default profiles maximize challenged user-space memory subject to reserve policy and honest attacker-budget accounting across unchallenged local tiers.
10. Production mode contains no silent downgrades, hidden stage copies, or undeclared shortcut paths.

---

## 26. Design invariants

The following invariants must remain true unless this spec is revised:

1. Paper-faithful protocol first.
   The production protocol is the graph-based PoSE-DB protocol described by the bundled paper.

2. One seeded labelling phase plus timed single-label rounds.
   The protocol is not split into separate proof layers.

3. Verifier/prover separation.
   The verifier remains a distinct process with distinct responsibilities.

4. No silent downgrades.
   Production mode must never silently fall back to untimed verification, managed memory, hidden stage copies, or weaker claim scopes.

5. Coverage honesty.
   All reported coverage must reflect actual challengeable slots in leased user-space regions.

6. Attacker-budget honesty.
   All unchallenged local memory usable during the fast phase counts toward the reported attacker budget unless separately constrained and documented.

7. Reference before acceleration.
   No accelerated component may replace the reference path without parity gates.

---

## 27. Open implementation notes

This section is informative.

- The initial implementation should optimize for clarity and claim discipline before micro-optimization.
- The strongest theorem-level claims will come from profiles that challenge as much relevant local memory as possible across all local tiers available to the attacker.
- Because fast rounds are timing-sensitive, same-host profiles should prefer a low-jitter IPC transport for the fast phase.
- The benchmark harness should preserve enough metadata that performance history is scientifically useful rather than anecdotal.
- Small exhaustive graph checks are far more valuable than large unverified graph dumps.
- Implicit graph generation is not an optimization detail; for realistic `m`, it is part of making the protocol implementable.

---

## 28. Summary

The repository shall be a Python-first PoSE-DB system with:

- the bundled depth-robust-graph protocol as its normative core,
- a clean prover/verifier process split,
- verifier-leased host and HBM regions,
- a canonical graph descriptor and label-array layout,
- timed single-label challenge rounds,
- explicit `q < γ` calibration,
- honest attacker-budget and claim-scope reporting,
- strong parity, benchmark, and hardware-test infrastructure,
- and an explicit path from development hardware to single-H100 and 8×H100 deployments.

That is the required shape of the implementation.
