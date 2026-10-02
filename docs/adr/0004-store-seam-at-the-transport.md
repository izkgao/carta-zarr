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
`ReadNodeMetadata` consults consolidated metadata before the transport, so a store that consolidated
its metadata is discovered without reading a document per variable. The `not_found` branch the
profile used to feed was unreachable and has been deleted.

Consolidated metadata is a copy that saves those reads, not a substitute for the documents it
copies. zarr-python writes it with `must_understand: false`, which says a reader may ignore it and
still read the same hierarchy, and it never removes the children when it writes one. This library
could not honour the other reading anyway: the array data behind those names is opened by
TensorStore from each array's own metadata, so a store whose children exist only in the root
describes images that nothing — this library included — can read pixels from. Such a store is
malformed, and the error it earns names the document that is missing. What the saving is worth is
measurable: in a release build, on a local filesystem with the page cache warm, consolidating an
11-variable XRADIO dataset takes `Dataset::Open` from 0.73 ms to 0.13 ms. Both numbers are metadata
end to end — a sampling profile of the consolidated open puts over half of it in parsing the root
document, a seventh in freeing it again, an eighth in turning it into array metadata, and two per
cent in the read itself — so what the copy saves is a real fraction of a real cost rather than a
rounding error. They are still under a
millisecond either way, and that is not the reason to keep it: it is that each of those reads
becomes one network round trip on a remote transport, which is what
consolidated metadata was invented for.

The seam is partial, and honestly so. Everything a probe needs is metadata, so a probe runs entirely
in memory; a descriptor that reports coordinate values still wants a store on disk. That asymmetry
is visible in the file layout: the single TensorStore-touching read lives alone in
`src/zarr/value_reader.cc`, so a consumer needing metadata but not values links without TensorStore
at all.

Transport is a seam with two adapters, not one. The filesystem transport serves production; the
in-memory transport in `tests/support/` serves the schema profile tests. A third — HTTP or S3 — is
what a remote store would need, and it arrives without the profile changing -- but not, as
first written here, without the library changing. See "Reopened" below.

An in-memory transport makes hand-writing store metadata easy, which is in tension with ADR-0003's
rule that fixtures come from the pinned generator. The line: the in-memory transport serves negative
and structural cases only — missing fields, wrong types, mismatched axes, consolidated metadata that
disagrees with the directory tree. Positive conformance is settled by generator fixtures on disk.

It also counts the nodes it was asked for, which is how the consolidated-metadata test states what
the copy is for. Asserting the images that come back cannot distinguish a store that used the copy
from one that ignored it, because both reach the same answer; asserting that no child was read can.

## Reopened: should the seam stop naming a filesystem?

`ArrayDirectory` returns a `std::filesystem::path`, which is the one thing in this seam that is not
about bytes. Reviewing it again, the seam keeps that shape, and two things about it were wrong
rather than merely partial.

The first is a claim above. "A third — HTTP or S3 — arrives without the profile changing" is true of
the profile and false of the library. Describing an image reads the `frequency` and `time`
coordinate values, so a transport with no filesystem behind it can serve `Probe` and a discovery and
then fail every `OpenImage`. A store that can be listed and never opened is a worse answer than one
that is refused, and that is what such a transport would deliver today.

The move that fixes it is known: `ArrayDirectory` would hand up a kvstore spec rather than a path,
and an HTTP transport would supply `{"driver": "http"}` where the filesystem one supplies
`{"driver": "file"}`. It is not being made now, for two reasons. There is one adapter that can
answer it — the in-memory transport reports `unsupported_transport`, as it should — so the variation
would be hypothetical, and a seam built for a caller that does not exist is guessed rather than
designed. And `string_array.cc` reads fixed-length UTF-32 chunk bytes off the filesystem directly,
without TensorStore; carrying it across a kvstore seam means owning what this ADR already declined
to own. When a second transport is actually written, this is the paragraph to come back to: the work
is that function plus the string reader, and nothing above them.

The second was a real fault. The filesystem transport kept the location exactly as the consumer
spelled it, and `Store` made the result absolute on every array read instead — which resolves
against the working directory *at read time*, not at open time. An image opened from a relative
location therefore stopped being readable when the process changed directory, and the failure landed
on a pixel read long after the open that looked fine. The resolution now happens once, in
`NormalizeLocation`, so a transport answers with a location that does not depend on the caller's
working directory; that is now part of what `ArrayDirectory` promises. It also takes a
`weakly_canonical` off the per-read path, where it cost a few `lstat` calls per cursor step. That
second part was not measured and is not the reason for the change: against a chunk decode it is
almost certainly below this machine's noise floor. Correctness is the reason.

`TestAnOpenImageOutlivesTheWorkingDirectory` in `tests/pixel_read_test.cc` pins it, and it pins the
part that matters: it opens relatively, moves away, and *then* reads. Reading before moving away
passes either way, which is why the fault survived this long.

Alongside it, what a carta-zarr array is made of — zarr3 over a `file` kvstore — was written out in
two places, one of which also bypassed the array-handle table. `OpenZarrArray` in
`zarr/store_context.cc` is now the only place it is said, which is the same function a second
transport would change.
