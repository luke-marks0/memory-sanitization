# Upstream Sync

## Official Source

The authoritative upstream repository for the production inner proof path is:

- `https://github.com/filecoin-project/rust-fil-proofs.git`

This repository vendors the upstream workspace intact under
`vendor/rust-fil-proofs/`. The current pinned snapshot is recorded in
`vendor/UPSTREAM.lock`.

## Current Pinned Snapshot

- upstream repository URL:
  `https://github.com/filecoin-project/rust-fil-proofs.git`
- default branch at time of sync: `master`
- pinned commit: `8186131e6d61fb7cc70191f7ad003364be7f6b57`
- source commit date: `2026-03-09T10:47:53+01:00`
- sync date: `2026-03-20T07:43:06Z`
- local patch status: `clean`

The upstream repository does not publish a single repository-wide release tag at
this commit. Instead, the commit is pointed to by the relevant crate release
tags such as `filecoin-proofs-v19.1.0`, `storage-proofs-porep-v19.1.0`, and
related workspace tags recorded in `vendor/UPSTREAM.lock`.

## Sync Procedure

The repository-owned sync entrypoint is:

```bash
make sync-upstream
```

The sync script reads `vendor/UPSTREAM.lock`, fetches the exact pinned commit
from the official upstream remote, replaces `vendor/rust-fil-proofs/` with that
snapshot, and verifies the vendored tree hash.

## Integrity Checks

The vendored tree is protected by:

- `vendor/UPSTREAM.lock`, which records the exact upstream source and the
  expected tree hash;
- `scripts/check_upstream_integrity.py`, which verifies the lock file and the
  vendored tree contents;
- `scripts/check_banned_shortcuts.py`, which scans production code for banned
  fake or shortcut proof identifiers;
- CI in `.github/workflows/foundation-integrity.yml`.

## Upstream Rust Verification

The repository-owned entrypoint for validating the vendored upstream workspace
is:

```bash
make test-upstream-rust
```

That command runs `scripts/run_upstream_rust_tests.py`, which:

1. verifies `vendor/UPSTREAM.lock` and the vendored tree hash;
2. installs the upstream Rust toolchain pinned by
   `vendor/rust-fil-proofs/rust-toolchain` if `rustup` is available to manage
   it;
3. installs the Ubuntu build prerequisites used by the upstream workspace when
   it is running on an `apt`-based Linux host with sufficient privileges;
4. hydrates the minimal proof-parameter and SRS set needed by the current
   vendored `cargo test --workspace --all-targets` run into
   `/var/tmp/filecoin-proof-parameters` (or `$FIL_PROOFS_PARAMETER_CACHE` if
   overridden);
5. reruns the workspace tests if the first cargo pass surfaces an additional
   missing proof artifact present in the vendored manifests.

The proof artifacts are fetched by exact filename from
`https://proofs.filecoin.io/` and are verified against the BLAKE2b-based
digests recorded in the vendored upstream manifests before being kept in the
local parameter cache.

## Patch Policy

Direct edits under `vendor/rust-fil-proofs/` are forbidden by default.

If an emergency patch becomes necessary:

1. the patch must be applied through an explicit patch mechanism, not by
   untracked manual edits;
2. `vendor/UPSTREAM.lock` must reflect non-clean patch status;
3. this document must record the patch rationale and scope;
4. integrity checks must acknowledge the patch state explicitly;
5. parity and upstream tests must still pass.

## Current Patch Status

No local patches are currently applied to the vendored upstream tree.
