# Tuning constants are measured on other machines, not configured on them

`src/reduce/tuning.h` holds the numbers the reductions were tuned to. Four of them were measured on
one machine and depend on its caches and its core count rather than on what is being computed:

- `kLeastPixelsPerTask`, below which a task is not worth waking a worker for;
- `kSpectralPartialBudgetBytes` and `kHistogramPartialBudgetBytes`, what a spectral reduction's and a
  plane histogram's per-task partials may cost together;
- `kCubeAccumulatorCacheBytes`, what a one-pass cube histogram's per-task accumulators may cost
  together, which is the cap ADR 0005 is about.

A server that is not the one they were measured on may want other values. They stay constants, with
no runtime setting and no backend flag, and each can be replaced when the library is built --
`CARTA_ZARR_TUNING_OVERRIDES`, checked by `cmake/TuningOverrides.cmake` -- so that
`carta-zarr-bench` can measure another value on another machine. A build with an override is a
measurement: every row the bench writes says which overrides were in force, and nothing deployed is
built that way.

## Why not a setting

A setting is a promise that the right value depends on the deployment and that the person deploying
can find it. Neither holds yet. Whether the right value moves from machine to machine is the open
question, and the person who would set it -- an administrator choosing `--zarr_file_io_threads` and
`--zarr_cache_size` from a sweep report -- has no way to see what these do except the same
measurement this decision provides for. A flag would also outlive the evidence: once a backend
configuration names one, it cannot be taken away without breaking that configuration, even after
measurement shows one value serves every machine.

Nor are they computed from the hardware at start-up, which is the other way a value could follow
the machine. That needs a rule -- so many pixels per task per megabyte of cache, say -- and a rule
needs more than one machine's numbers to be told from a coincidence.

## Why only these four

The other constants in `tuning.h` decide what a reduction answers, not how fast it answers it.
`kProvisionalBinsPerBin` and its bounds set how finely a one-pass cube histogram resolves a bin, and
so where its edges land; `kSpectralEmitBudgetBytes` bounds how much one emitted block may hold, which
a caller's sink sees. Replacing either would change results, and a measurement that changes the
answer is not comparing like with like. The four that can be replaced only change how work is
divided, which two builds of the synthetic reduction tests check: one with every split taken and one
with none, both expected to give every answer the default build gives. Across the measurements below
the fingerprints of every position agreed between every variant as well.

## What decides whether one becomes machine-dependent

Evidence from at least two machines that differ in core count or cache, each showing the same
direction of gain, and a rule that predicts both. Until then the default stays, and the warm stage of
`tools/zarr-bench/sweep.py` -- which measures builds with overrides against the default one and never
recommends any -- is where that evidence is collected.

## Measurements

Warm (`--cold off`, one untimed pass first), on the 512 x 512 x 7776 ASKAP cube in its own layout, at
`--io-threads 8 --cache-bytes 1G`; region and the two cube histograms as `carta-zarr-bench` runs
them, three trials interleaved across the builds (ten for acdc's single-user exact histograms).
Medians against the default build of the time, which had `kLeastPixelsPerTask` at 65,536; positive
is faster.

| variant | users | exact histogram | | binned histogram | | region | |
|---|---|---|---|---|---|---|---|
| | | acdc | almat3 | acdc | almat3 | acdc | almat3 |
| default | 1 | 5.50 s | 6.24 s | 1.49 s | 1.55 s | 0.84 s | 1.06 s |
| `LEAST_PIXELS_PER_TASK=4096` | 1 | +15% | +23% | – | -3% | -1% | +1% |
| `LEAST_PIXELS_PER_TASK=16384` | 1 | +12% | +21% | +0% | -4% | -0% | +1% |
| `LEAST_PIXELS_PER_TASK=262144` | 1 | -21% | -41% | +6% | -4% | +1% | +3% |
| `LEAST_PIXELS_PER_TASK=1048576` | 1 | -20% | -42% | -8% | -29% | -3% | +1% |
| `CUBE_ACCUMULATOR_CACHE_BYTES=1048576` | 1 | +1% | +0% | -11% | -29% | +0% | +1% |
| `CUBE_ACCUMULATOR_CACHE_BYTES=4194304` | 1 | -0% | -0% | +2% | -1% | +8% | -0% |
| `CUBE_ACCUMULATOR_CACHE_BYTES=33554432` | 1 | +0% | +1% | +1% | -8% | +5% | -0% |
| every variant | 8 | within ±5% | within ±6% | within ±1% | within ±2% | within ±1% | within ±2% |

acdc is a desktop: an Intel i7-14700F, 28 logical cores, 28 MiB of L2 and 33 MiB of L3, with the cube
on local NVMe. almat3 is a server: two Xeon Gold 6134 sockets, 32 logical cores, 1 MiB of L2 a core
and 25 MiB of L3, with the cube on Lustre. Every position every build read fingerprinted the same on
both, 81 positions each.

With eight users the cores are busy with eight processes, and how each divides its work no longer
matters. With one, the exact histogram -- the backend's default -- gained on both machines from
smaller tasks and lost on both from larger ones. Its second pass bins one plane at a time, a 512 x 512
plane is 262,144 pixels, and at 65,536 a plane was four tasks however many cores there were.

`kCubeAccumulatorCacheBytes` was right where it was on both: half as much cost the binned histogram
11% and 29%, and more gained nothing.

## What the measurements decided

Two machines that differ in nearly everything agreed on the direction, so `kLeastPixelsPerTask` is
now 16,384 -- a better constant for every machine, which is not the same thing as a machine-dependent
one. 4,096 gained a few percent more on both, less than it risks on a machine whose dispatch is
slower, for a read too small to be worth splitting that finely. Nothing here argues for computing it
from the hardware yet: no value was best on one machine and worse than the default on the other.
