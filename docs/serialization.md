# Canonical Serialization

## Scope

This repository now defines a canonical Python-owned serialization contract for
one PoRep unit.

A serialized PoRep unit is:

1. a deterministic-CBOR manifest;
2. followed by concatenated blob payloads in fixed manifest order;
3. followed by zero-byte alignment padding so the next unit starts on a leaf
   boundary.

This is the canonical byte format for the object stored in challenged memory.

## Manifest

The manifest records:

- format name and version;
- protocol version;
- upstream snapshot identifier;
- proof config identifier;
- storage profile;
- sector size;
- registered seal proof and API version;
- `porep_id`, prover id, sector id, ticket, and seed;
- piece information;
- `comm_d` and `comm_r`;
- inner proof timings;
- leaf alignment bytes;
- payload length bytes;
- fixed-order blob table with offsets, lengths, encodings, and SHA-256 digests.

The current implementation lives in
`src/pose/filecoin/porep_unit.py`.

## Blob Order

Blob kinds are ordered exactly as follows:

1. `seal_proof`
2. `sealed_replica`
3. `tree_c`
4. `tree_r_last`
5. `persistent_aux`
6. `temporary_aux`
7. `labels`
8. `cache_file`
9. `public_inputs`
10. `proof_metadata`

Unknown blob kinds are rejected.

## Profiles

The serializer supports the spec-defined profiles:

- `minimal`
- `replica`
- `full-cache`

Today, the real bridge can automatically materialize `minimal` units from a real
seal artifact. `replica` and `full-cache` are supported by the serializer when
their additional blob payloads are supplied explicitly.

## Region Placement

The PoRep-unit manifest does not decide where a unit is stored.

Region selection, lease handles, host-vs-GPU placement, and session manifest
roots remain part of the later prover/verifier and outer-proof layers.
