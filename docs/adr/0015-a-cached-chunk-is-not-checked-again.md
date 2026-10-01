# A cached chunk is not checked against storage again

**Status: proposed, measured, not adopted.** This branch holds the change and its test; `dev` does not
take it until a measurement shows a gain. See the end.

Every array this library reads is opened with TensorStore's `recheck_cached_data` set to `"open"`: a
chunk decoded after the array was opened is used from the cache as it is, without asking storage
whether its file has changed. TensorStore's default, `true`, asks on every read.

## What asking cost

Asking is not free even when the answer is "unchanged". TensorStore validates a cached chunk by
opening its file and comparing what it finds with what it cached, so a read the cache could answer
still opens one file per chunk. On a local disk that is a system call. On a parallel file system it
is a round trip to the metadata server, the one resource every client of the file system shares, and
it is paid by exactly the reads a cache exists to make cheap: an animation whose frames share chunks,
a plane revisited, a profile redrawn.

It also meant a cached chunk was only as available as its file. A chunk the cache held could not be
read once its file was unreadable, which is how this was found: a test that made a dataset's chunks
unreadable after reading them could not tell a cached read from a fresh one, because both failed.

## Why "open" and not false

The two differ only for data cached before the array was opened, and carta-zarr opens an array before
anything is cached for it -- each `CachePool` opens its arrays afresh -- so in practice they behave the
same. `"open"` keeps the narrower promise, and is the setting TensorStore already uses for metadata
by default.

## What it gives up

A dataset rewritten at the same path while a session has it open may be read partly from the cache
and partly from the new files for the rest of that session. That was already true of its metadata,
which TensorStore does not recheck after opening either, and of the opened arrays this library keeps
for the life of a context. A CARTA session does not write the data it views, and rewriting a dataset
under a running viewer was never something a read could rely on noticing.

`tests/store_context_test.cc` pins the behaviour: once a copy's chunks are truncated, the handle that
read them reads the same values again, and a handle opened afresh cannot read them at all.

## Measurements, and why it waits

On almat3's Lustre 2.15 -- one client, an idle metadata server -- with every data cache emptied before
each run by `tools/zarr-bench/drop-lustre-cache.py`, one user, on a 2048-channel crop of the ASKAP cube
in three layouts. Medians of the run before the change and after it, six runs each, alternating which
ran first and each preceded by an untimed run of its own:

| layout | animation, per frame | region |
|---|---|---|
| 512 x 512 x 1 | 5.8 ms, 5.8 ms | 810 ms, 824 ms |
| 512 x 512 x 16 | 4.3 ms, 4.2 ms | 781 ms, 792 ms |
| 128 x 128 x 128 | 3.3 ms, 3.8 ms | 99 ms, 99 ms |

With eight users the two were within 4% in every layout and mode. There is no gain to see here.

A first measurement showed one of 50% to 70%, and it was the order of the runs: whichever build ran
second after the data caches were emptied was faster. What carries over is not known. It is not the
files' metadata -- opening every file first changed nothing -- nor any read an ordinary user can make
before the run: only a run of the bench itself evened it out. It holds for any comparison of builds or
settings on Lustre, which is why the sweep now runs the bench once, untimed, on every dataset.

The cost of the check may yet show where this measurement could not see it: many clients on one
metadata server, under load. Until it does, the change gives up the guarantee for nothing measured,
and `dev` keeps TensorStore's default.
