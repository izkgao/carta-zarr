# Tuning Zarr storage for CARTA on Lustre and BeeGFS

For administrators of a CARTA server whose image cubes are XRADIO Zarr datasets on a parallel file
system. It says what can be tuned, what measurements on two file systems found, and how to measure
your own. The tools themselves are described in
[`tools/zarr-bench/README.md`](../tools/zarr-bench/README.md).

## What can be tuned

Two things, by two different people at two different times:

- **The layout of the data**, chosen when a cube is converted to Zarr: the chunk shape, whether
  chunks are grouped into shards, the compressor, and on Lustre the striping of the directory it is
  written to. Changing it means rewriting the cube.
- **carta-backend's reader settings**, chosen when the server is configured, as flags or in
  `settings.json`:

  | flag | default | what it sets |
  |---|---|---|
  | `--zarr_file_io_threads` | 2 | how many reads of chunk files run at once |
  | `--zarr_data_copy_threads` | every logical core | how many threads decode chunks |
  | `--zarr_cache_size` | 1024 (MiB) | the decoded-chunk cache each backend process keeps |
  | `--zarr_histogram_method` | `exact` | how a cube histogram is computed |

  carta-controller starts one backend process per user, so every user gets these, and a server with
  sixteen users at once has sixteen caches.

The constants inside carta-zarr's reductions are not settings; [ADR 0014](adr/0014-tuning-constants-are-measured-not-configured.md)
explains why, and how they are measured.

## What the measurements found

On almat3 (two Xeon Gold 6134, 32 logical cores, 1 TB RAM) against its Lustre 2.15 and BeeGFS file
systems, with eight users reading at once and every cache emptied before each trial, and on acdc
(a 28-core desktop) against a local NVMe disk for comparison. The numbers below are medians per
operation; a plane read is what jumping to another channel costs, since carta-backend reads the whole
plane of a Zarr cube into memory and downsamples it there.

### The best chunk depth depends on the size of the plane

A chunk deep in channels holds a long run of each pixel's spectrum, so a spectrum reads few chunks;
but a plane then reads every channel of its chunks, not only its own. Which matters more depends on
how large the plane is.

A 512 x 512 x 4096 crop of an ASKAP cube on Lustre:

| chunk (l x m x channels) | plane | spectrum | region (5% of the plane) |
|---|---|---|---|
| 512 x 512 x 1 (the cube's own) | 6 ms | 2.40 s | 6.02 s |
| 512 x 512 x 16 | 37 ms | 2.36 s | 3.54 s |
| 256 x 256 x 32 | 41 ms | 529 ms | 1.38 s |
| 128 x 128 x 128 | 146 ms | 222 ms | 382 ms |

A synthetic 2048 x 2048 x 2048 cube on the same Lustre:

| chunk (l x m x channels) | plane | spectrum | region |
|---|---|---|---|
| 2048 x 2048 x 1 | 205 ms | 14.8 s | 28.1 s |
| 512 x 512 x 16 | 478 ms | 1.55 s | 2.93 s |
| 256 x 256 x 32 | 1.08 s | 838 ms | 4.12 s |
| 128 x 128 x 128 | 2.81 s | 246 ms | 4.04 s |

On the 512-square plane eight users' spectra and regions could not all read chunks of their own --
there are too few chunks across it -- so those columns include some reads the page cache answered,
which if anything flatters the layouts with large chunks across the sky.

With every mode weighted alike, 128 x 128 x 128 served the 512-square cube best, and 512 x 512 x 16
the 2048-square one, where the same 128-deep chunks made a plane take nearly three seconds. A chunk
one plane deep -- what a cube converted plane by plane gets -- gives the fastest channel changes and
the slowest spectra by a factor of ten or more. Which to prefer is a judgement about how a site's
users work; the sweep's `[weights]` makes it explicit, and its report says when doubling the weight of
planes or of spectra would change the answer. For the 512-square cube it would: weighting planes
twice as heavily brings the cube's own layout back to first place.

**A rule of thumb.** Each operation took time roughly in proportion to the pixels it decoded: a plane
decodes the plane's area times the chunk depth, a spectrum the chunk's area times the number of
channels. For chunks of about the same size the two balance at a depth of about

    depth ≈ sqrt(chunk pixels × channels / plane pixels)

which gives about 180 for the 512-square cube (128 measured best) and 30 to 45 for the 2048-square one
(16 to 32 measured best), with chunks of 2 to 4 million pixels. It is a starting point for the grid a
sweep measures, not an answer: regions, animations and cube histograms pull in other directions, and
it was fitted to two cubes.

These sweeps tried chunks of 8 to 16 MiB of float32. Chunks of 4 MiB, a million pixels -- 512 x 512 x 4
or 256 x 256 x 16, which CARTA's own tests have mostly used -- were not among them. By the rule, a
million-pixel chunk suits a 2048-square, 2048-channel cube at about 22 channels deep, close to
256 x 256 x 16, and a 512-square, 4096-channel one at about 126, far deeper than 512 x 512 x 4. Put the
chunk shapes a site already uses into its sweep's grid, so that the report compares every candidate
against them.

### Use more file-reading threads on Lustre

`--zarr_file_io_threads 8` was better than the default of 2 on Lustre in every sweep: a plane of the
128-deep layout went from 146 ms to 104 ms and a spectrum from 222 ms to 157 ms; a region of the
cube's own layout from 6.35 s to 3.33 s; a plane of the 2048-square cube's 512 x 512 x 16 layout from
478 ms to 358 ms. On BeeGFS the difference was within the noise.
`--zarr_data_copy_threads` at half the logical cores was as good as all of them.

### Striping and shards

Striping a dataset's directory across one OST or across eight made no difference beyond 5% for chunks
of a few megabytes: a chunk file smaller than the stripe size lives on one OST either way. What
striping cannot change is the number of files, which every open and every stat sends to the metadata
server that all clients share: a 32 GiB cube in 128-deep chunks is 4,000 files, and a terabyte cube
in the same chunks would be over 100,000. Shards -- chunks gathered into one file, each still read on
its own -- are the remedy, and are worth including in a sweep on a busy file system. The sweep
accepts them relative to the chunk, `frequency*8` or `l*4,m*4`.

## Measuring your own system

The answer on another file system, another cube size or another number of users can differ, and the
tools exist so that it can be measured rather than assumed.

1. **Build** carta-zarr in Release with `-DCARTA_ZARR_BUILD_BENCH=ON`; a Debug build is several times
   slower and the report says so.
2. **Write a configuration** from [`example-sweep.toml`](../tools/zarr-bench/example-sweep.toml):
   the bench, a work directory on the file system to measure, a real cube to rewrite (or a synthetic
   shape), the number of users at the busiest time and on an ordinary day, the layouts to try, and the
   reader settings to try on the best of them. If the site holds cubes of very different sizes, list
   each as a `[[source.shape]]`.
3. **Check it:** `./sweep.py my-sweep.toml --dry-run` says what would run, how much space it needs,
   how caches will be emptied, and whether striping can be set, and writes nothing.
4. **Run it:** `./sweep.py my-sweep.toml`. It writes, measures and deletes one layout at a time, and
   the same command resumes it after an interruption. A sweep of four layouts of a 32 GiB cube took
   about two hours on almat3's Lustre.
5. **Read `summary.md`**: the layout and the flags to use, as they would be pasted into the backend's
   command line or `settings.json`, then every warning, then the tables behind them.

### Empty the right caches

A read that finds its data in memory measures memory. Between a CARTA server and the disks there are
at least three caches: carta-zarr's own (each read in the bench starts without it), the client's page
cache, and the storage servers'. On almat3's Lustre a whole-cube read took 1.24 s with everything
cached, 2.0 s with the client's page cache emptied, and 5.5 s with the servers' emptied too -- but
11.9 s the first time after the cube was written. Something below what an ordinary user can empty
keeps data that was read recently, and no later read was as slow as the first.

The sweep therefore runs the bench once, untimed, every time it writes a dataset, before measuring
it. Every layout is then measured in the same state, which is what a comparison needs, and its times
are those of data read recently rather than data never read: on almat3 a third to half faster for a
region of the cube's own layout. A first view of a cube nobody has opened in a while will be slower
than the report says.

- With root, `drop_caches` empties the client's page cache; give the sweep a `drop_cache_cmd` that
  empties the servers' too if the storage administrators provide one.
- Without root on Lustre,
  [`drop-lustre-cache.py`](../tools/zarr-bench/drop-lustre-cache.py) empties both: the client's by
  `posix_fadvise`, and the servers' by `lfs ladvise -a dontneed`, which Lustre 2.9 and later accept
  from a file's owner.
- Without root on BeeGFS nothing reaches the storage servers' caches, and the report says reads were
  cold on this host only. Layouts measured second can then look faster than they are.

### Things that mislead a measurement

- **Too shallow a crop.** A plane or a spectrum is timed as the first read of its chunks. A crop of
  1,024 channels holds only eight chunks 128 deep, and sixteen users each reading sixteen planes
  share them, so most reads find their chunks already in the page cache. The report counts the reads
  that did not, ranks on those, and warns when there are too few, with what to change.
- **The order of runs on Lustre.** With the caches emptied the same way before each, the first run
  on a dataset after it was written, or after other datasets were read, was slower than the ones that
  followed it: a region of a cube in one-channel chunks took 1.2 s to 1.7 s the first time and 0.8 s
  every time after. Opening every file first, reading a byte of each, reading all of them, or waiting
  two minutes did not change it; only a run of the bench did. A comparison of two builds once showed
  a 50% to 70% gain that vanished when the order was alternated. The sweep's untimed run puts every
  measured run after one; compare builds or settings by hand the same way.
- **One user is not eight.** Settings that help one user can do nothing for eight, whose processes
  already keep every core busy; the report gives the target number of users, and how the choice does
  with one and with the typical number.
- **Data that fits in memory.** A crop smaller than RAM measures caching unless caches are emptied
  before every trial; the sweep refuses one when they cannot be. Validation re-measures the
  recommendation on a copy larger than RAM, which is impossible on a server with as much memory as
  almat3's.
- **Synthetic pixels.** A layout's cost depends on how well its pixels compress, and noise compresses
  worse than sky. Rewrite a real cube when there is one.

## Very large planes

Cubes from the SKA or the ngVLA may be tens of thousands of pixels on a side and hundreds of channels
deep. Two things change.

**Every channel change reads the whole plane.** carta-backend reads the plane of a Zarr cube in full
and downsamples it for display, so a 30,000 x 30,000 plane is 3.6 GB of float32 per channel change,
however shallow the chunks. No layout makes that interactive; what would is a downsampled copy beside
the full one, which carta-backend does not read yet. Until it does, the rule of thumb above points to
chunks one channel deep for such a plane, and a spectrum then reads a chunk per channel.

**A layout cannot be written at full depth for every try.** A sweep shape may keep the full plane and
fewer channels, with `channels` saying how deep the cubes it stands for are:

```toml
[[source.shape]]
name = "ska"
synthetic = "frequency=64,polarization=1,l=32768,m=32768"
channels = 512
```

Planes and animations are then measured at their full size, and the summary scales the modes that
read every channel to the full depth, as an estimate. A chunk deeper than the shape's channels is
skipped, since it would not be the chunk a full cube has.

## Where the numbers came from

The sweeps behind this page ran on 2026-10-01 and 2026-10-02 with carta-zarr at the commits that
added each tool, eight users, two or three trials, and every cache emptied as described above. Their
configurations and full reports were kept on the machines they ran on rather than in this
repository, since they describe those machines; rerun the sweep on yours rather than reuse them.
