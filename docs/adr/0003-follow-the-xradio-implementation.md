# Attribute layouts follow the XRADIO implementation, not its schema document

Where the XRADIO schema document and the released XRADIO code disagree about how an attribute is
spelled or where it lives, this library follows the code. The schema document is an unfinished draft
of a v2 layout; the code is what produced every file we will actually be handed.

Two places where the two differ today, all of which we resolve toward the code:

- The direction coordinate lives in the root `coordinate_system_info` attribute, not in the
  `attrs.direction` the document describes.
- An image's role is on the data variable's own `type` attribute, lowercased (`"sky"`, `"model"`,
  `"residual"`). No released XRADIO writes the `image_type` attribute the document specifies for
  FITS `BTYPE`.

## Consequences

Reading an attribute the document specifies but no XRADIO version writes is not a harmless
precaution: it produces a field that is always empty, and it hides the fact that the real value was
somewhere else. Every attribute this library reads must be traceable to a line of XRADIO that writes
it. Where a documented v2 name is worth accepting for forward compatibility, it is accepted as a
fallback behind the name the implementation actually uses, never in front of it.

Detection is based on the image variables and their coordinates. A store must contain at least one
complete, real-valued sky-plane image, and malformed coordinate metadata is reported as invalid.

Fixtures must come from the pinned generator, never by hand. A hand-written fixture encodes the same
assumption as the code that reads it, so the pair agrees with itself and disagrees with reality — a
test suite in that state reports success while the library cannot open a real file.
