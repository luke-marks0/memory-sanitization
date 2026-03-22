# Result Schema

## Verdicts

The target stable verdict set is:

- `SUCCESS`
- `WRONG_RESPONSE`
- `DEADLINE_MISS`
- `CALIBRATION_INVALID`
- `COVERAGE_BELOW_THRESHOLD`
- `RESOURCE_FAILURE`
- `CLEANUP_FAILURE`
- `PROTOCOL_ERROR`

## Required Fields

The canonical result artifact is the full serialized `SessionResult` object.
`pose verifier verify-record` treats that schema strictly: every field emitted by
the repository is required, even when its value is empty or zero.

The required fields are:

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
- `run_class`
- `session_seed_commitment`
- `artifact_path`
- `resident_socket_path`
- `resident_process_id`
- `lease_expiry`
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
- `scratch_peak_bytes`
- `declared_stage_copy_bytes`
- `round_trip_p50_us`
- `round_trip_p95_us`
- `round_trip_p99_us`
- `max_round_trip_us`
- `cleanup_status`
- `formal_claim_notes`
- `operational_claim_notes`
- `claim_notes`
- `timings_ms`
- `environment`
- `notes`

## Timing Breakdown

The timing artifact should cover at least:

- discovery;
- region leasing and allocation;
- graph construction;
- challenge schedule preparation;
- expected-response preparation;
- label generation;
- copy to host/HBM where applicable;
- stage-buffer cleanup;
- fast-phase total;
- verifier check total;
- cleanup;
- total runtime.
