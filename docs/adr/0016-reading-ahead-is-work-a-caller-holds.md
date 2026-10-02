# Reading ahead is work a caller holds

`ReadAhead` decodes the next run of chunks of a playing animation on a thread of its own, while the
frames of this run play from the cache. It is the one place this library does work outside a call a
caller is making, and it does it only inside an object the caller holds: the thread belongs to the
`ReadAhead`, at most one decode is under way on it at a time, and the last copy of the handle to go
cancels that decode and waits for it. Nothing it started outlives it.

## Why the library and not each caller

The decisions are about chunks and caches, which is what this library knows and its callers do not.
Which planes share the chunks a frame decodes, how much those chunks occupy once decoded -- a flag
decodes beside the pixels at a byte an element -- and how much the cache they go into holds are all
questions an `Image` answers. carta-backend answered them for itself, through its own reading of the
chunk shape and its own copy of the decoded-size rule, and the benchmark answered them a third way:
it modelled the backend's reading ahead and had drifted from it, deciding runs along the spectrum
alone and never asking whether the cache could hold what it decoded.

What a caller does know is when a frame was played, whether it was late by the caller's own
measure, and which planes come next. That is what `ReadAhead::Served` is told, after every frame.

## Why its own thread

A decode ahead is started by one frame and waited for by a later one, so it has to run on something
other than the thread playing the frames. The library's `WorkPool` is the wrong thing: a prefetch
spends its time waiting on decodes that are themselves spread over the decode threads, and a worker
of the pool held for that is a worker the reductions it shares the pool with do not get. A caller
supplying its own executor would have been an interface for a variation nobody has -- both callers
started a thread.

## Why only when there is time over

Measured on almat3's Lustre with a 7763 x 4742 cube, a frame that enters a run of chunks decodes all
of it: in 512 x 512 x 4 chunks it stalled for up to 135 ms every fourth frame at 5 frames a second,
and in 512 x 512 x 16 chunks for up to 400 ms every sixteenth, while every other frame was served
from the cache in a few milliseconds. Decoding the next run while the frames of this one played hid
every stall for one viewer, at 5 and at 10 frames a second.

With eight viewers animating that cube at once, prefetches that kept ahead of every run still
doubled the time of the frames played from the cache, and at 10 frames a second made more of them
late however soon they stopped. So once a frame is late while a prefetch is under way, there are no
more for the rest of the animation: the machine has no time over, and decoding ahead takes it from
the frames being played. A frame that catches a prefetch it was waiting on is not one of those -- it
waits for the decode under way rather than starting its own, which is what ADR 0015 makes safe.

And only when each cache the animated images read through holds two runs of every image reading
through it: the one playing and the one decoded ahead. Otherwise the run decoded ahead evicts the one
being played. A run of 512 x 512 x 4 chunks of that cube is 589 MB; of 512 x 512 x 16, 2.4 GB.

How far ahead to look -- 64 frames, which at CARTA's 5 frames a second is 13 s before the next run is
needed -- is a constant of the library's, as ADR 0014 has it, and published so that a caller builds
no more of what is to come than is looked at.

## What it gives up

A caller can no longer read ahead by a policy of its own through this object; it can still call
`Image::Prefetch`, which remains the primitive underneath. The thread is started per prefetch rather
than kept, which costs a thread start a run of chunks -- tens of microseconds against decodes of
hundreds of milliseconds.
