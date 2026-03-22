# Graph Construction

## Scope

This document describes the repository's target graph construction model for the
PoSE-DB protocol defined in `repository-spec.md`.

The normative source of truth remains the bundled paper plus the repository
spec. This file exists to make the implementation work concrete and auditable.

## Required Graph Family

The production graph family is:

- `pose-db-drg-v1`

That identifier binds:

- arbitrary-`m` graph construction;
- deterministic node numbering;
- deterministic predecessor ordering;
- deterministic ordered challenge set `O(G)`;
- deterministic graph descriptor encoding.

## Required Construction Rules

For a target label count `m`, the implementation must:

1. choose the smallest integer `n` such that `2^(n+1) >= m`;
2. construct the corresponding depth-robust graph implicitly;
3. expose an ordered challenge set of size exactly `m`;
4. report `gamma = 2^n`.

Production implementations must keep the graph implicit for large sessions.
Fully materialized graphs are acceptable only for small exhaustive tests and
debugging.

## Descriptor And Ordering

The graph descriptor must be encoded in deterministic CBOR and hashed to produce
`graph_descriptor_digest`.

The descriptor must bind at minimum:

- graph family ID;
- `m`;
- `n`;
- `gamma`;
- node-ordering version;
- challenge-set ordering version;
- hash backend;
- `w_bits`.

The implementation must define deterministic rules for:

- topological node numbering;
- predecessor ordering;
- challenge-set ordering.

## Labels

The covered state for a session is the label array:

`sigma = l(o1) || l(o2) || ... || l(om)`

Each challenge index refers to one position in the ordered challenge set and
therefore one physical slot in one verifier-leased region.

The exact encoding of node IDs, predecessor count, and label inputs must be
fixed in code and parity-tested.

## Implementation Notes

The Python reference implementation should live in:

- `src/pose/graphs/`
- `src/pose/hashing/`

Any native acceleration must be parity-gated against those semantics before it
is eligible for production use.
