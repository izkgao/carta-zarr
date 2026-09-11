# The store seam sits at the transport, and coordinate value reads sit outside it

A `Store` reads a Zarr hierarchy on behalf of a schema profile. The seam between "what the bytes
mean" and "where the bytes are" is placed at the **transport**: a `Transport` hands up one node's
`zarr.json` verbatim, lists the nodes, and resolves a node's array path. Everything that turns those
bytes into meaning — node path validation, JSON parsing, consolidated metadata, array metadata,
storage layout, and the caches — stays in `Store`, above the seam. Coordinate value reads are
deliberately left outside the seam: they take a filesystem path, and a transport that has none
reports `unsupported_transport`.

## Considered options

Making `Store` itself the interface, with a filesystem implementation and an in-memory one, was the
obvious alternative and was rejected. Under it the in-memory implementation reimplements
consolidated-metadata collapse and array-metadata parsing, so the two implementations can disagree
about precisely the metadata handling the tests exist to pin down. Placing the seam lower means the
interpretation of a store has exactly one implementation by construction.

Carrying coordinate values across the seam was considered twice. Serving them through TensorStore's
own `memory` kvstore is cheap to wire up — `tensorstore::kvstore_memory` is already available — but
it makes a test fixture a blob of real Zarr chunk bytes, which is more setup than writing files, not
less. Decoding numeric chunks ourselves, the way the `fixed_length_utf32` reader already does for
string coordinates, would remove TensorStore from the metadata path entirely, but it means owning
numeric codec decoding for no present gain.

## Consequences

The profile no longer knows what a filesystem is. It previously reached through `Store` to build
`root / "SKY" / "zarr.json"` and stat it directly, which was not equivalent to asking the `Store`:
`ReadNodeMetadata` consults consolidated metadata before the filesystem, so a store whose child
metadata exists only in consolidated metadata is still discovered. The `not_found` branch it used
to feed was unreachable and has been deleted.

The seam is partial, and honestly so. Everything a probe needs is metadata, so a probe runs entirely
in memory; a descriptor that reports coordinate values still wants a store on disk. That asymmetry
is visible in the file layout: the single TensorStore-touching read lives alone in
`src/zarr/value_reader.cc`, so a consumer needing metadata but not values links without TensorStore
at all.

Transport is a seam with two adapters, not one. The filesystem transport serves production; the
in-memory transport in `tests/support/` serves the schema profile tests. A third — HTTP or S3 — is
what `docs/design.md` §4 anticipates, and it arrives without the profile changing.

An in-memory transport makes hand-writing store metadata easy, which is in tension with ADR-0003's
rule that fixtures come from the pinned generator. The line: the in-memory transport serves negative
and structural cases only — missing fields, wrong types, mismatched axes, consolidated metadata that
disagrees with the directory tree. Positive conformance is settled by generator fixtures on disk.
