# The pass inlines its visitor, and has one accumulator per task

A **pass** is one ordered visit to every chunk an image read covers, shared by every reduction that
wants those pixels. Two things about its shape look like accidents of implementation and are not,
so they are recorded here: the visitor is a template parameter rather than a virtual interface or a
`std::function`, and there is one accumulator per task, with the split capped by what the
accumulators cost rather than by how many workers the pool has.

Both are the kind of thing a later reading would tidy up. A pass that took a `Visitor&` interface
would be easier to describe, and a split that used every worker in the pool would look like the
point of having one. Each of those changes costs a measured amount of time, and neither cost is
visible in a Debug build or in any test this repository runs.

## The visitor is a template parameter

The per-pixel loop inlines through the visitor. `AccumulateRow` is instantiated four ways on
`(unit stride, masked)`, chosen once per row, and the histogram's `Add` and its row lambdas reach
the same state by being local types behind a template parameter. Type-erasing the visitor puts an
indirect call at the slab boundary and stops the enclosing loop nest from inlining.

`54731c1`, which made the accumulation loop vectorise, is worth about 25% of a whole-region
reduction: a 240 x 240 x 250 profile settles at 24-28 ms against 33 ms, with the reads unchanged at
10.7 ms. Its own message records what makes this dangerous: "That figure is from a Release build.
The same comparison in the default Debug build dir shows no difference at all, which is how this
change was nearly discarded as useless."

`WorkPool::Run` does take a `std::function`, and that is not in tension with this: it is paid once
per task, outside the pixel loop, not once per slab inside it.

## There is one accumulator per task, and never more tasks than accumulators

A provisional histogram is half a megabyte of scattered writes, and every worker streams its share
of the pixels through the same cache. Past a couple of megabytes of them the pixels evict the
histograms and the pass gets slower the more workers it uses. Measured on a 512x512x7776 ASKAP cube,
warm, against 8.8 s for not splitting at all: four workers 4.9 s, eight 8.7 s, twenty-eight 16.5 s.

So what bounds the split is cache rather than the pool: the cap is a memory budget divided by what
one accumulator costs, which at the default resolution comes out at four. The budget is
`kCubeAccumulatorCacheBytes` in `tuning.h`, beside the two the other reductions divide in the same
way; theirs bound an allocation rather than cache, and are not this cap. `Accumulator` is
`alignas(64)` because `Add` writes the range and a bin on every pixel, and two accumulators sharing
a cache line would trade it between cores once per pixel.

The accumulator a body writes is chosen by **task** index, and it is safe because the split never
asks for more tasks than there are accumulators, so no two bodies ever hold the same one at once.
That guarantee has one home, `TaskSplit` in `reduce/task_split.h`: it works out the cap from the
budget and the accumulator's size, never answers more tasks than that, and is the only way a
reduction reaches the pool.

This is worth stating precisely because the pool offers the other choice and nothing uses it.
`WorkPool::Run` hands the body a worker index as well as a task index, exactly so that "a caller
that needs private accumulation can address one slot per worker without a map or a lock" -- and
`TaskSplit::Run`, the one call site all three reductions go through, takes `(std::size_t task,
std::size_t)` and ignores it. Since tasks are
claimed from a shared counter rather than divided up front, task n is run by a different thread on
each slab, so a task-keyed accumulator does migrate between cores as the pass advances. Whether
keying by worker instead would recover anything is unmeasured, and is not settled here; what is
settled is that the two are different and that the code does the first.

## Image::Read stays outside the pass

The pass was built to collapse three walks over a cube into one. Two of them collapsed. The third,
`Image::Read`, was migrated last precisely so that it would be the test of the shape, and it failed
that test, which is the answer rather than a disappointment.

`ReadSlab` reverses `logical_to_stored` so a plane arrives in the order the store wrote it and is
never transposed; the visitor pays for that in strides it was going to walk anyway. `Image::Read`
has to do the opposite, because its destination is the caller's buffer and its contract is a dense
image in logical order. Beyond that it splits along the slowest selected axis of an arbitrary
request rather than along u, v and the spectrum; it reads the flag *before* the pixels, so that an
unavailable mask cannot leave a piece of the caller's destination updated, where the pass reads
pixels first; and it reports progress in elements rather than in chunks. Fitting it would take a
second entry point or a handful of parameters that do nothing for anybody else.

What it shares it already shared: `ReadCost`, `DefaultReadBytes`, `ChunksTouched` and
`AlignedBlockEnd` in `chunk_blocks.h`. `UnitsPerPiece` does not duplicate that policy -- it
converts the byte budget the policy returns into a count of elements along one axis, which is a
thing the pass never needs.

They do now agree on what a masked read costs, which they did not when the pass was written. The
pass doubled its budget when it would also read a flag and `Image::Read` ignored the flag entirely,
so one overstated the cost and the other understated it. Neither figure was right: `RequireUsableFlag`
holds a flag to `bool` over the image's own shape, so beside a float32 chunk it is a quarter of one.
`ReadCost::Of` answers it and both sides call it -- in `Image::Read`'s case in both halves of its
sizing, the budget and the per-row cost that budget is divided by, because counting it in one and
not the other sizes pieces against a cost the read does not have.

A quarter of a chunk was still an equal-chunk answer: it took the flag to be chunked as its pixels
were, and ADR 0016 records that it need not be. Both walks are handed the flag's own geometry now,
and `DecodedFlagBytes` charges each pixel chunk the flag chunks it lies across, whole. Chunked alike
or finer, that is the same byte an element; coarser, it is more, and for a flag kept in one chunk it
is all of it. A read of several pixel chunks sharing one flag chunk decodes that chunk once, so this
over-counts such a read -- deliberately: both walks count in pixel chunks, an over-count costs a read
some parallelism, and an under-count spends memory the caller's budget said not to.

What a chunk costs is what it holds while it is read, not only what it decodes to, and both walks
keep to the budget whether or not anybody watches; `Image::Read` reads a piece too large at one
chunk deep in segments. ADR 0021 records why.

## Consequences

The pass cannot be given a non-template entry point for convenience, and a second overload taking a
`std::function` would be a trap rather than a shortcut: it would compile, pass every test, and cost
a quarter of the reduction. If one is ever wanted for a cold path, it belongs behind a name that
says so. The same holds for `TaskSplit::Run`, which takes its body as a template parameter and
calls it directly when a read is one task.

The numbers above cannot be reproduced by anything in this repository. Every fixture is far too
small to show either effect, no test asserts a duration, and the default `build/` tree is Debug, so
the loop change measures as nothing there. A timing harness run by hand against `build-release/` is
what stands in for a regression test, and it is the only thing that does.

Neither decision reaches the public interface. `src/reduce/` is entirely
`carta::zarr::internal`, and only `include/carta-zarr/` is installed.

Collapsing the three walks into one pass preserves the task keying rather than settling it. Changing
it in the same step would move a number the collapse is being measured by, and there is no
measurement either way to move it toward.
