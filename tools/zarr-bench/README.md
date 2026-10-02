# zarr-bench

Tools for finding which Zarr layout, and which carta-zarr settings, read fastest on a given storage
system -- Lustre and BeeGFS in particular, where the answer differs from a local disk.
[`docs/storage-tuning.md`](../../docs/storage-tuning.md) says what they have found so far and how to
read what they report; this page is how to run them.

## generate.py

Writes one XRADIO image dataset in a chosen layout. It is a self-contained
[`uv`](https://docs.astral.sh/uv/) script, so it needs nothing installed but `uv`.

Rewrite a real cube, cropped to something that can be rewritten many times over:

```sh
./generate.py --source /data/cube.zarr --crop frequency=0:2000 \
    --chunk l=512,m=512,frequency=16 --shard frequency=256 --codec blosc:zstd:5:shuffle \
    --output-root /lustre/scratch/zarr-bench
```

Or synthesize one, when there is no real cube to hand:

```sh
./generate.py --synthetic --shape frequency=4000,polarization=1,l=4096,m=4096 \
    --chunk l=256,m=256,frequency=64 --codec zstd:3 --stripe lustre:count=4,size=4M \
    --output-root /lustre/scratch/zarr-bench
```

- **Axes** are named as XRADIO names them -- `time`, `frequency`, `polarization`, `l`, `m` -- and an
  axis left out of `--chunk` is 1. An axis left out of `--shard` is its chunk.
- **Codecs:** `none`, `zstd[:level]`, `gzip[:level]`, `blosc[:cname[:level[:shuffle]]]`.
  `--keep-bits N` rounds pixels to N mantissa bits before compressing them.
- **Real pixels are the better source.** What a layout costs depends on how well the pixels
  compress, and synthetic noise does not compress like a real image. The synthetic cube is
  deterministic: two layouts of the same `--seed` and `--shape` hold the same pixels.
- **`--stripe`** sets Lustre (`lfs setstripe`) or BeeGFS (`beegfs-ctl --setpattern`, or
  `beegfs entry set` on BeeGFS 8) striping on the output directory before anything is written,
  since striping applies only to files created after it. BeeGFS normally reserves this for root;
  the generator stops rather than writing an unstriped dataset.
- **`--output-root`** names the dataset after a hash of everything that decides its bytes, and
  reuses one that is already complete. `--output` names it yourself.

- **Memory:** each worker holds about three times what it writes at once -- `--block-mib`, or one
  whole shard when shards are larger than that. Lower `--workers` for large shards.
- **Python:** the script runs on uv's own Python build, never the system's. Ubuntu 24.04's 3.12.3
  segfaults as soon as zarr starts its event-loop thread.

Each dataset carries `bench-manifest.json`: its source, layout, striping as the filesystem reports
it, file count and compression ratio. The manifest is written last, so a dataset without one is
unfinished. The last line on stdout is the dataset's path.

- **`--layout-from-source`** takes the chunk, shard, codec and consolidation from the source rather
  than from `--chunk`, `--shard`, `--codec` and `--no-consolidate`: the source's own layout over the
  crop, to compare every other layout with. sweep.py calls it `current`.

`ctest` checks the generator against the library when configured with
`-DCARTA_ZARR_BUILD_BENCH=ON` and `uv` is on the path.

## carta-zarr-bench

Measures how fast the library reads one dataset the way CARTA reads it. Built with
`-DCARTA_ZARR_BUILD_BENCH=ON`, as `bench/carta-zarr-bench` in the build tree; measure with a Release
build, since a Debug one is five to seven times slower and says so on every run.

```sh
carta-zarr-bench probe /lustre/scratch/zarr-bench/<dataset>
carta-zarr-bench run /lustre/scratch/zarr-bench/<dataset> --processes 8 \
    --io-threads 16 --decode-threads 8 --cache-bytes 2G --csv results.csv --resume
```

`probe` prints what the library sees of a dataset as one line of JSON, and fails when it would not
open. `run` writes one CSV row per operation; `--help` lists its options.

- **Modes.** `plane` reads a whole plane at a random channel, `animation` plays
  `--animation-frames` consecutive planes (32) from a random channel at `--animation-fps` frames a
  second (5, as CARTA's animator does), recording which frames were late and by how much, `spectrum` every channel at a
  random pixel, `region` reduces every statistic over a box covering `--region-fraction` of the plane
  (5%), `cube-histogram` bins the cube, and `open` times `Context::Create`, `Dataset::Open` and
  `OpenImage` together. `--ops 8,spectrum=64` sets how many operations each makes per trial.
- **First touches.** `plane` and `spectrum` read each operation through a cache pool of its own, made
  before the clock starts and the size of `--cache-bytes` (1 GiB for `default`, the backend's own
  default). Otherwise a layout whose chunks are 128 channels deep would answer most of a trial's
  planes from the cache the first one filled, and its median would be a cache hit. `animation` is
  the mode where that reuse is real -- a user playing through channels -- and reads through the
  context's shared cache.
- **`shares_chunks`** marks an operation that reads a chunk an earlier operation of its process read,
  or one any other process reads. Its time may be the page cache's, so it is not a first touch however
  fresh its cache pool; sweep.py ranks `plane`, `spectrum` and `region` without such operations.
- **Cube histograms** are computed as carta-backend computes them by default
  (`--zarr_histogram_method exact`): a reduction over whole planes for the range, then every plane
  binned over it -- two passes over the cube, through a cache pool that keeps nothing, as the
  backend's are. `--histogram-method binned` or `sampled:N` times `ComputeCubeHistogram`'s one pass
  instead.
- **Positions** come from `--seed` and the trial number and the cube's shape, never its layout, so
  every layout of a cube is read at the same places. They are not aligned to chunks.
- **Processes.** `--processes N` stands for N users, since carta-controller starts one backend per
  user: N processes, each with its own context, released together. They read different positions;
  a cube histogram splits the channels between them. When there are not enough positions to go
  round, the rows that share one say `overlap`.
- **Trials.** Each trial empties the caches, then forks fresh processes, so no trial inherits a cache
  or a thread pool from the one before. Within a trial a process keeps its context, as a backend
  would: a region or an animation may find chunks an earlier operation decoded, and an `open` after
  the first finds the metadata in the page cache.
- **Cold reads.** `--cold auto` uses `--drop-cache-cmd` when given, then `drop_caches` as root, then
  `posix_fadvise` on every file of the dataset. Only the command can reach a parallel filesystem's
  servers; the other two empty this host's page cache alone, and `cold_method` says which was used.
  On macOS only the command is available. On Lustre, [`drop-lustre-cache.py`](#without-root) is such
  a command that needs no root.
- **Deadline.** `--trial-timeout` becomes each read's `ReadControl::deadline`. An operation it stops,
  and any not yet started, is a `timeout` row; a process still running 30 s past it is killed.
- **Resuming.** Rows are written a trial at a time, so an interrupted run loses at most the trial it
  was in. `--resume` skips the trials the CSV already holds for the same settings -- the `run_key`
  column -- that finished without an error.
- **`checksum`** fingerprints what each operation returned, taken after the clock stops: the pixels
  for a read, the pixel counts and extremes for a reduction or a histogram, and an exact histogram's
  counts, which every layout of the same pixels must agree on exactly. Sums are left out, because
  their rounding follows the order the chunks were visited in, and so are a one-pass histogram's
  counts, which depend on the thread count.
- **`storage_read_bytes`** is what `/proc/self/io` says the operation fetched from storage, so it is
  empty off Linux.

## sweep.py

Runs the other two over every layout and setting worth trying, and says which to use. Copy
[`example-sweep.toml`](example-sweep.toml), whose every field is explained, and:

```sh
./sweep.py my-sweep.toml --dry-run     # what would run, the disk it needs, and whether it can
./sweep.py my-sweep.toml               # run it; the same command resumes it
./sweep.py my-sweep.toml --set users.target=32 --output results-32
```

It works in four stages, each written into one `results.csv` under its own label:

1. **stage1** writes each layout in turn -- the grid of `[stage1]` and the source's own, `current` --
   measures it at the baseline settings with one user and with `users.target`, and deletes it.
2. **stage2** takes the best layouts of stage 1, and `current`, and tries every reader setting of
   `[stage2]` on them with the target number of users.
3. **confirm** measures the recommended settings, their nearest rivals and the baseline with one
   user and with `users.typical`: what choosing for the peak costs everyone else.
4. **validate** writes the recommended layout again from `validate_crop`, larger than RAM, and checks
   that what the smaller copy said still holds. It needs caches that can be emptied.
5. **warm**, only when `[warm]` lists builds of the bench with [tuning overrides](#tuning-constants),
   measures each against the default build on the recommended layout and settings, with the data in
   the page cache. It is reported and never recommended.

`summary.md` opens with the settings to give carta-backend -- as flags and as `settings.json` -- for
the data as it is, and for once it is in the recommended layout, and every warning: layouts that
failed, cold reads that were cold on this host only, a mode the recommendation gives up, results
that did not survive validation. The tables behind it follow.

- **Cube shapes.** The best layout depends on the size of the plane: on a 512 x 512 cube chunks 128
  channels deep served spectra best at little cost to planes, and on a 2048 x 2048 one the same
  chunks made a plane take seconds. `[[source.shape]]` sweeps each shape a site holds on its own,
  and the summary in the output directory compares them and says what one layout and one setting --
  the backend reads every cube with the same flags -- would serve them all. A shape too large to
  rewrite at its full depth for every layout, as an SKA or ngVLA plane tens of thousands of pixels
  square would be, is measured on fewer channels with `channels` saying how many it stands for: its
  planes are measured at full size, and the modes that cover every channel are scaled to the full
  depth, as an estimate. A layout whose chunk is larger than a shape's cube is skipped for that
  shape, since it would not be the chunk a full cube has.
- **Shards** can be said relative to each chunk, `frequency*8` or `l*4,m*4`, so that one list fits
  every chunk of the grid. On Lustre a shard turns many chunk files into one, which spares the
  metadata server.
- **Choosing across modes.** Each mode is ranked on its own, by the median operation -- for `plane`,
  `spectrum` and `region` the median first touch, and for `animation` the median frame. Where one
  choice has to serve them all -- the layouts stage 2 tries, the settings recommended -- each mode's
  slowdown against its own best is combined in a geometric mean weighted by `[weights]`, and the
  largest slowdown is reported beside it.
- **Chunk depth** trades planes against spectra: deeper chunks read a spectrum in fewer pieces and
  make a plane read the channels around it too. Stage 1 ends with a table of each layout's side of
  that trade, the layouts no other beats in every mode marked as the front, and a note when doubling
  the weight of `spectrum` or of `plane` would change the best layout.
- **Too few first touches.** When a crop is shallow for its chunk depth -- 1024 channels hold only 8
  chunks 128 deep, against 16 planes a user -- fewer than five operations of a mode read chunks of
  their own. That mode is then ranked on every operation, and the report says so and what to change.
- **The read budget** has no backend setting, so a setting with one is never recommended. The report
  says when one would make a mode at least 10% faster, which is the case for adding the setting.
- **Resuming.** `sweep-state.json` records each finished run, so a rerun skips it without writing its
  layout again; a run cut short resumes from the CSV. Runs that failed are tried again.
- **`--report-only`** rewrites `summary.md` from what there is, mid-sweep or after.

The output directory, `zarr-bench-results` beside the config unless `--output` says otherwise, holds
`results.csv`, `summary.md`, `sweep-state.json`, `config.toml` (the configuration as run, every
default filled in), `machine.json`, and a log per stage and layout under `logs/`.

## Tuning constants

A few constants of the library's reductions -- how many pixels a task needs, and what the tasks'
private accumulators may cost together -- were measured on one machine and depend on its caches and
core count. They are not settings: the backend has no flag for them, and ADR 0014 records why. They
can be replaced when the library is built, for measuring another value on another machine:

```sh
cmake -S carta-zarr -B build-tuning -DCMAKE_BUILD_TYPE=Release -DCARTA_ZARR_BUILD_BENCH=ON \
    -DCARTA_ZARR_BUILD_TESTS=OFF
for variant in cube-4m:CUBE_ACCUMULATOR_CACHE_BYTES=4194304 tasks-16k:LEAST_PIXELS_PER_TASK=16384; do
    cmake build-tuning -DCARTA_ZARR_TUNING_OVERRIDES="${variant#*:}"
    cmake --build build-tuning --target carta_zarr_bench -j
    cmake --install build-tuning --prefix "/opt/zarr-bench-variants/${variant%%:*}"
done
```

`CARTA_ZARR_TUNING_OVERRIDES` takes `NAME=VALUE` pairs separated by `;`, for `LEAST_PIXELS_PER_TASK`,
`SPECTRAL_PARTIAL_BUDGET_BYTES`, `HISTOGRAM_PARTIAL_BUDGET_BYTES` and
`CUBE_ACCUMULATOR_CACHE_BYTES`. Changing it rebuilds the library and not TensorStore, so a variant
takes minutes. Each goes to a prefix of its own, because the bench finds its library beside it: a
bench run against another build's library would write the wrong overrides into its rows. Every row
has a `tuning` column saying which were in force, `default` for none.

List the variants in `[warm]` and the sweep measures them against `paths.bench`, which must be built
without overrides. The constants left out decide what a reduction answers rather than how fast, and
cannot be replaced.

## Without root

Everything here runs as an ordinary user: `uv` installs into the home directory, the bench builds
anywhere a C++17 compiler and CMake 3.24 are (on an old distribution, a conda-forge `gxx` and
`python>=3.10`, which TensorStore's build needs, will do), and its processes need no privilege. What
root buys is emptying caches, and on a parallel file system the caches that matter most are on the
servers.

- **Lustre.** [`drop-lustre-cache.py`](drop-lustre-cache.py) `DIR` empties this client's page cache
  for every file under `DIR`, as `--cold fadvise` would, and then asks the servers to drop theirs
  with `lfs ladvise -a dontneed`, which Lustre 2.9 and later accept from a file's owner. Give it the
  sweep's work directory, which holds one dataset at a time:

  ```toml
  [measure]
  cold = "command"
  drop_cache_cmd = "/opt/carta-zarr/tools/zarr-bench/drop-lustre-cache.py /lustre/scratch/zarr-bench"
  ```

  Without the second step a sweep measures the servers' memory, not their disks: on one Lustre 2.15
  system a whole-cube read took 1.2 s warm, 2.0 s with only the client's cache emptied, and 5.5 s
  with both. `lfs setstripe` on a directory of one's own needs no root either, so `stage1.stripe`
  can be swept.
- **BeeGFS.** Nothing an ordinary user can call empties a storage server's cache, so `--cold fadvise`
  is the most there is, and the report says reads were cold on this host only: a layout read twice
  in a row reads faster the second time. A client whose `tuneFileCacheType` is `buffered` keeps
  little in its own page cache to begin with. `beegfs-ctl --setpattern` is root's unless the
  file system is configured to let users set it; `--dry-run` tries it, and says when
  `stage1.stripe` cannot be swept.
- **Quotas.** sweep.py checks the file system's free space, not a user's quota. Check that
  `lfs quota -u $USER` or `beegfs-ctl --getquota --uid $USER` leaves room for one uncompressed crop
  and a tenth more.
