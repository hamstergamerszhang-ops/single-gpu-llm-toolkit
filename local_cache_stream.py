#!/usr/bin/env python3
"""Durably materialize any generator to a local JSONL cache, then stream rows
back out of it -- zero network dependency once the cache exists.

Genericized from two real, project-specific scripts:
  - materialize_cpt_cache.py (the write side): pulled rows from a live
    HF-streaming data pipeline and wrote them incrementally to /dev/shm,
    because a full training run outlives the network being reliably up, and
    losing the whole capture to one mid-run failure was a real risk with a
    "hold it all in a list, write once at the end" approach.
  - cpt_streaming_data.py's stream_cpt_rows_from_cache() (the read side):
    loads the finished cache into memory once, shuffles it with a seed, and
    yields rows in a loop, reshuffling on every full pass so a long run
    doesn't see the exact same row order repeat forever.

The reusable idea underneath both, stripped of anything HF/dataset-specific:
any process that produces dict rows from a generator (a live stream, an API
with a flaky connection, a slow multi-hour ETL) can be captured to durable
local disk incrementally, and re-read later as an infinite, reshuffled-per-pass
iterator -- useful any time your data SOURCE is less reliable than your
TRAINING RUN needs to be, which is common on a single unattended GPU box
where nobody is present to babysit a stalled network connection.

Usage as a library:
    from local_cache_stream import materialize_to_cache, stream_from_cache

    def my_source():
        # any generator yielding dicts, e.g. wrapping a flaky API/stream
        for row in some_upstream_iterator:
            yield {"text": row["text"]}

    materialize_to_cache(my_source(), "./cache/data.jsonl", target_rows=500_000)

    for row in stream_from_cache("./cache/data.jsonl", seed=42):
        train_on(row)

Self-test (no network, no GPU/model required -- exercises both functions
against an in-memory fake generator and a real temp-dir JSONL file):
    python3 local_cache_stream.py --selftest
"""

import argparse
import json
import os
from pathlib import Path


def materialize_to_cache(row_generator, dst_path: str, target_rows: int = 500_000,
                          flush_every: int = 2000) -> int:
    """Write rows from `row_generator` (any iterator yielding JSON-serializable
    dicts) incrementally to `dst_path` as JSONL, one line per row.

    Incremental + periodic flush (not "collect a big list, then write once at
    the end") means a large capture is safe to interrupt -- a crash or a
    manual stop after row 300,000 of a 500,000-row target still leaves a
    fully usable partial cache on disk, not nothing.

    If `row_generator` raises partway through (e.g. the underlying source
    goes unreachable), this stops early and returns however many rows it
    managed to capture rather than propagating the exception -- a partial
    cache is more useful than a failed run producing zero rows. Genuinely
    fatal problems (out of disk, permission denied) DO propagate; only the
    generator's own exceptions are caught, since "the source ran dry or
    errored" is the expected failure mode this function exists to survive.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    total_bytes = 0
    print(f"[local_cache_stream] target: {target_rows:,} rows -> {dst_path}", flush=True)

    with open(dst_path, "w", encoding="utf-8") as f:
        while n_rows < target_rows:
            try:
                row = next(row_generator)
            except StopIteration:
                print(f"[local_cache_stream] source generator exhausted at {n_rows:,} rows "
                      f"(target was {target_rows:,}) -- stopping, cache is still usable",
                      flush=True)
                break
            except Exception as e:
                print(f"[local_cache_stream] source generator raised ({e}) -- stopping "
                      f"early with {n_rows:,} rows captured, still usable", flush=True)
                break
            line = json.dumps(row) + "\n"
            f.write(line)
            total_bytes += len(line.encode("utf-8"))
            n_rows += 1
            if n_rows % flush_every == 0:
                f.flush()
                print(f"[local_cache_stream] {n_rows:,} rows, {total_bytes/1e9:.3f} GB so far",
                      flush=True)

    print(f"[local_cache_stream] done: {n_rows:,} rows ({total_bytes/1e9:.3f} GB) -> {dst_path}",
          flush=True)
    return n_rows


def stream_from_cache(cache_path: str, seed: int = 42):
    """Infinite generator reading rows back out of a JSONL cache built by
    materialize_to_cache().

    Memory-efficient approach: scans the file ONCE to record byte offsets of
    each valid line, then seeks+reads individual lines on demand. Only the
    active row is in RAM at a time — a multi-GB cache no longer needs to fit
    in memory (the old version loaded the whole file into a Python list of
    dicts, ~3-5x the on-disk size due to object overhead). The offset index
    is just a list of ints (8 bytes per row), so even a 10M-row cache's
    index is only ~80MB.

    Shuffles the offset index with the given seed, then yields rows in a
    loop, reshuffling on each full pass so a very long run doesn't see the
    exact same row order repeat forever.

    Raises RuntimeError immediately if the cache file is empty -- training
    against zero rows is a silent no-op you want to fail loudly on, not a
    generator that just never yields anything.

    Malformed lines (e.g. a truncated last line from a crash mid-write) are
    skipped with a warning rather than crashing the streamer.
    """
    import random

    # Pass 1: record byte offsets of each valid line. Only the offsets (ints)
    # are held in RAM, not the row contents.
    offsets = []
    malformed = 0
    with open(cache_path, encoding="utf-8") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            line = line.strip()
            if line:
                try:
                    json.loads(line)  # validate without storing
                    offsets.append(offset)
                except json.JSONDecodeError:
                    malformed += 1
            offset = f.tell()
            line = f.readline()
    if malformed:
        print(f"[local_cache_stream] WARNING: skipped {malformed} malformed "
              f"line(s) in {cache_path} (likely a truncated last line from a "
              f"crash mid-write). {len(offsets):,} valid rows indexed.",
              flush=True)
    if not offsets:
        raise RuntimeError(f"Cache at {cache_path} is empty -- nothing to stream")

    print(f"[local_cache_stream] indexed {len(offsets):,} row offsets from "
          f"{cache_path} (memory-mapped access, only active row in RAM)",
          flush=True)

    # Pass 2+: seek+read individual lines on demand, in shuffled order.
    rng = random.Random(seed)
    pass_num = 0
    f = open(cache_path, encoding="utf-8")  # kept open for the lifetime of the generator
    try:
        while True:
            order = list(range(len(offsets)))
            rng.shuffle(order)
            pass_num += 1
            if pass_num > 1:
                print(f"[local_cache_stream] pass {pass_num}: reshuffled and looping over the "
                      f"same {len(offsets):,} cached rows again (cache exhausted, repetition beats "
                      f"stopping)", flush=True)
            for idx in order:
                f.seek(offsets[idx])
                line = f.readline().strip()
                yield json.loads(line)
    finally:
        f.close()


def _self_test():
    import tempfile

    print("[selftest] materialize_to_cache() writes exactly target_rows from a finite "
          "in-memory generator, incrementally, as JSONL")
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "cache.jsonl")

        def fake_source(n):
            for i in range(n):
                yield {"text": f"row-{i}"}

        n_written = materialize_to_cache(fake_source(50), dst, target_rows=50, flush_every=10)
        assert n_written == 50, n_written
        with open(dst, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 50
        assert lines[0] == {"text": "row-0"}
        assert lines[49] == {"text": "row-49"}
        print("  OK (50/50 rows on disk, order preserved on write)")

        print("[selftest] materialize_to_cache() stops early and cleanly when the "
              "generator raises, keeping whatever it already captured")
        def flaky_source():
            yield {"text": "ok-1"}
            yield {"text": "ok-2"}
            raise ConnectionError("simulated network drop")

        dst2 = os.path.join(td, "cache2.jsonl")
        n_written2 = materialize_to_cache(flaky_source(), dst2, target_rows=100)
        assert n_written2 == 2, n_written2
        with open(dst2, encoding="utf-8") as f:
            lines2 = [json.loads(l) for l in f if l.strip()]
        assert len(lines2) == 2
        print("  OK (partial cache preserved, exception did not propagate)")

        print("[selftest] materialize_to_cache() stops cleanly when the generator "
              "exhausts before target_rows (StopIteration, not an error)")
        dst3 = os.path.join(td, "cache3.jsonl")
        n_written3 = materialize_to_cache(fake_source(5), dst3, target_rows=100)
        assert n_written3 == 5, n_written3
        print("  OK")

        print("[selftest] stream_from_cache() yields every row at least once per pass, "
              "and reshuffles across passes")
        gen = stream_from_cache(dst, seed=7)
        first_pass = [next(gen) for _ in range(50)]
        assert {r["text"] for r in first_pass} == {f"row-{i}" for i in range(50)}
        second_pass = [next(gen) for _ in range(50)]
        assert {r["text"] for r in second_pass} == {f"row-{i}" for i in range(50)}
        # Overwhelmingly likely to differ in order across a reshuffle of 50 items;
        # not a hard guarantee (rng CAN produce the same order), but this is the same
        # style of probabilistic check as train_cpt.py's own selftest.
        print(f"  OK (pass 1 order == pass 2 order: {first_pass == second_pass} -- "
              f"expected False with overwhelming probability on 50 items)")

        print("[selftest] stream_from_cache() raises RuntimeError on an empty cache "
              "instead of silently yielding nothing")
        empty_path = os.path.join(td, "empty.jsonl")
        Path(empty_path).write_text("")
        try:
            next(stream_from_cache(empty_path))
            raise AssertionError("expected RuntimeError on empty cache")
        except RuntimeError:
            print("  OK")

    print("\n[selftest] All checks passed (no network or GPU required -- these exercise "
          "only local generators and a temp-dir JSONL file).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    args = ap.parse_args()
    if args.selftest:
        _self_test()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
