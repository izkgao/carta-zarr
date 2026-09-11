# Images are identified by variable name and discovered by inspection, not by data group

An XRADIO image dataset declares its images twice: implicitly, as data variables carrying the `l`
and `m` coordinates, and explicitly, in the optional root attribute `data_groups`, which maps roles
such as `sky` and `residual` to variable names. We identify an image by its data-variable name
alone, and we discover the set of images by inspecting every variable in the dataset rather than by
reading `data_groups`.

## Considered options

Identifying an image by a `(data group, role)` pair was the alternative, and it is what the XRADIO
schema is built around. We rejected it because CARTA presents images in a flat dropdown, the way it
presents FITS HDUs, and because a variable shared by two data groups would appear twice under that
scheme. Variable names are unique within a dataset, so the name alone is sufficient to address one.

Discovering images from `data_groups` was rejected on evidence rather than on principle: the
XRADIO implementation writes `data_groups` (defaulting to `{"base": {}}`) but populates it sparsely,
commonly with no more than `{"base": {"sky": "SKY"}}`, so a dataset containing a `RESIDUAL` would
report only its `SKY`. Inspection reads what is actually in the store.

## Consequences

An image is a data variable that carries the whole sky axis set (`time`, `frequency`,
`polarization`, `l`, `m`), holds a real data type, and whose `type` attribute is not `"flag"`.
Matching the whole set rather than just `l` and `m` is not pedantry: XRADIO writes `right_ascension`
and `declination` by default as float64 arrays over `(l, m)` carrying no `type` attribute at all, so
a rule keyed on the spatial pair alone offers them to the user as openable images. It also excludes
the `(time, frequency, polarization)` normalization variables and the beam fit parameters, and it
means the optional coordinates are never read. Two further consequences may surprise a reader:

- `MASK_DECONVOLVE` is an image, because it satisfies every condition. It is displayed like any
  other image.
- `MASK_0`, despite its name, is not an image. The XRADIO CASA reader names variables after CASA's
  internal masks but tags them `type: "flag"`, so they are pixel masks. Classification must never be
  driven by variable names.

A known variable-name list is still kept, but only to order the images for display so that `SKY`
appears first. It never decides membership.

`data_groups` is carried into `ImageDescriptor` as descriptive metadata. It is deliberately not used
to locate an image's flag or beam variable, because those associations are written to the image
variable's own attributes today and to `data_groups` only in the not-yet-implemented v2 schema.
