# gemma-mi300x-prune-cpt

Ten real, independently-runnable tools for adapting an LLM checkpoint on a
**single AMD GPU** under ROCm/PyTorch — no multi-node cluster, no
distributed training framework. They came out of actually doing this once,
for real: shrinking a tokenizer, growing a model, and continue-pretraining
it, all on one MI300X, and then hitting the specific ways a single GPU
fails you (OOM, crashes, a data source that goes unreachable mid-run) and
fixing each one for real instead of writing around it. Every script here is
real, run code, not a from-scratch rewrite for this repo — project-specific
naming genericized, everything else unchanged.

Everything below was only ever actually run against one setup: a Gemma-4-
family checkpoint, one MI300X. But every model-family-specific assumption
that used to be a hardcoded constant — the embedding tensor's key name, the
vocab_size config path, the layer-naming prefix, the sharding size, the
depth/width step sizes — is now a CLI flag with that real setup as the
default, not something you'd need to fork the source to change. That's a
narrower claim than "works on anything," and it's the honest one: this is
*configurable* toward other model families and left as an open question
whether it's *correct* there, since nobody's pointed it at a non-Gemma
checkpoint yet. If you do, the flags exist for exactly that, and the
README calls out per-tool which specific assumptions are still Gemma-4-only
(mainly the GQA fix and a couple of `expand_model.py`'s submodule key
suffixes) versus genuinely architecture-agnostic already.

You don't need to use these together or in order — each one solves a
different single-GPU problem on its own:

## `prune_vocab.py` — shrink a tokenizer you don't need in full

A Gemma-4 tokenizer ships vocabulary for scripts you may never see — CJK,
Cyrillic, Arabic, Devanagari, Mongolian, a long tail of accented Latin. If
your use case doesn't need all of that, `prune_vocab.py` drops those entries
by a configurable character-script heuristic, remaps every surviving token
to a contiguous new ID space, and filters the BPE merge table to match. One
detail that mattered in practice: some Gemma-4 configs store `vocab_size` in
two places, a top-level field and a nested one, and missing either one
silently reverts the vocab size at load time — this script fixes both, and
it fixes both because a real load crashed on exactly this the first time
around. Useful on its own any time you want a smaller embedding table
without retraining the tokenizer from scratch.

```
python3 prune_vocab.py --src <base_checkpoint> --dst <pruned_checkpoint>
```

## `prune_embeddings_torch.py` — apply that vocab cut to the actual weights

Dropping tokenizer entries doesn't shrink anything until the model's actual
weights follow. This script takes the ID remap the tool above produces and
slices `embed_tokens.weight` down to match, handling both sharded (with an
`index.json`) and single-file checkpoints — it rewrites only the shard that
changed and copies the rest through untouched, rather than reserializing
weights it didn't need to touch. Useful standalone any time you've already
got a vocab remap and just need the tensor surgery.

```
python3 prune_embeddings_torch.py --src <base_checkpoint> --dst <pruned_checkpoint>
```

## `expand_model.py` — grow a model's width and depth without retraining from scratch

The opposite problem: instead of shrinking a checkpoint, grow it. This
widens the MLP intermediate dimension and duplicates decoder layers to
increase parameter count from an existing checkpoint, and it uses two
different init strategies depending on what's actually being added. New
width columns get an **orthogonal-QR init**, because they need a real,
non-conflicting gradient signal from step one — zero-init would leave them
starved. Newly duplicated layers get **zero-init on their output
projections only**, which makes the insertion a true no-op: the layer runs
a real forward pass, but contributes nothing to the residual stream until
training turns it on. There's also an optional GQA fix for full-attention
layers that ship with a single shared KV head and no separate `v_proj` —
worth applying when KV-cache size isn't your actual memory bottleneck. Pure
PyTorch, no Apple-Silicon-only dependency, usable on any checkpoint you want
to grow rather than retrain.

```
python3 expand_model.py --src <pruned_checkpoint> --dst <expanded_checkpoint>
```

**AMD-specific note:** it uses `numpy.linalg.qr`, not `torch.linalg.qr`, for
the orthogonal constructions — this ROCm PyTorch build has no LAPACK support
for CPU tensors, so `torch.linalg.qr` on CPU raises directly, not
approximately, not sometimes. Swap it if your build has working CPU-tensor
QR; this repo keeps numpy because it's what actually ran.

## `train_cpt.py` — continued pretraining, single GPU

This is the actual CUDA/ROCm training loop, and the rest of this README is
mostly about the problems it ran into and how they got fixed: layer-window
freeze/unfreeze (full-model training when VRAM allows, partial-layer
windowing when it doesn't), gradient checkpointing, an 8-bit-Adam-with-AdamW-
fallback optimizer, async local-disk checkpointing, a local-JSONL data/cache
mode, and a clean SIGTERM-triggered checkpoint-and-exit. It's the standalone
entry point for training any checkpoint — pruned, expanded, or neither.

```
python3 train_cpt.py --model <checkpoint> --data <jsonl_dir_or_file> --save <out_dir> --batch 1
```

As of this pass, four pieces that used to live inline inside `train_cpt.py`'s
`main()` — optimizer construction, async checkpoint writes, the
optimizer-type resume guard, and local-cache data streaming — are their own
standalone modules now (see "Standalone utilities" below). `train_cpt.py`
imports and calls them rather than duplicating the logic, and its own
`--selftest` still passes with the same "no torch/GPU required" guarantee it
always had.

## `catch_and_resume.sh` — keep a single-GPU run alive across crashes

`train_cpt.py` already self-resumes on its own — it checks for
`<save_dir>/training_state.pt` on startup, no `--resume` flag needed, just
re-run the same command and it picks up where it left off. What self-resume
alone doesn't give you is judgment about *whether* the checkpoint it's about
to resume from is actually good, and that's what this wraps around it: a
**loss-tagged checkpoint history** with rollback if the latest checkpoint's
loss spiked above the best one kept so far, a **bounded retry** for crashes
that keep happening at the same position (so a genuinely recurring bug
doesn't just retry silently forever, eating GPU-hours on a loop that was
never going to succeed), and a **stop-file** for requesting a clean shutdown
between attempts instead of having to reach for `kill -9`.

```
./catch_and_resume.sh
```

## Standalone utilities

Three of these came directly out of the user's ask for the parts of
`train_cpt.py` that were doing real, non-trivial work but only existed as
prose and inline logic buried in `main()` — worth pulling out on their own
merits, not just for tidiness. A fourth is a port of a memory-safety script
that started life solving a Mac-specific crash but whose actual pattern —
poll, warn, kill before the OS does something worse — has nothing
Mac-specific about it. A fifth closes a small but genuinely nasty gap: what
happens when you resume training with a *different* optimizer than the one
that saved the checkpoint.

**`bnb_optimizer.py`** exists because "which optimizer did this run
actually get" turns out to matter a lot on a single GPU, and it's not a
question you want answered differently by two copies of the same
try/except scattered across two scripts. It tries bitsandbytes' 8-bit Adam
first — both first and second moment buffers at roughly 1 byte/param
instead of fp32 AdamW's 4 bytes/param, which is the difference between
comfortably fitting a large model plus its optimizer state on an 80GB+ card
and being one missing pip install away from an OOM. If bitsandbytes isn't
importable, it falls back to plain `torch.optim.AdamW` with an explicit
warning, and — this is the part worth calling out — the fallback's failure
mode isn't a crash at step 0. It's an OOM dozens of iterations in, once the
roughly 4x-larger optimizer state has actually finished allocating across
all the trainable params. That delay is exactly what makes it confusing to
debug if you don't already know to check for a silently-missing
bitsandbytes install first. `build_optimizer(model, lr, weight_decay)`
returns both the optimizer and which kind it built, so a caller can log the
decision instead of discovering it three OOMs later.

```python
from bnb_optimizer import build_optimizer
optimizer, kind = build_optimizer(trainable_params, lr=8e-7, weight_decay=0.01)
```

**`async_checkpoint.py`** is the background-thread checkpoint writer,
pulled out of `train_cpt.py` where it used to be a ~100-line class buried
inside a 33KB training script. The idea is straightforward once it's
isolated: serializing tens of GB to a possibly-slow disk or NFS mount is
slow, and there's no reason the GPU should sit idle waiting for it. So the
class splits the work into two phases — a synchronous GPU-to-CPU snapshot
(brief, and it has to be synchronous, because the GPU tensors are about to
be mutated by the very next training step), followed by an asynchronous
disk write that only ever touches the CPU copy and is safe to run
concurrently with several more training steps. It's bounded to one in-flight
write at a time — `save()` will block on any still-running previous write
before starting a new snapshot — which trades an occasional wait for a hard
guarantee against unbounded CPU-RAM growth if writes ever fall behind the
checkpoint interval. It writes to local disk only, atomically (temp
directory, then rename), and getting checkpoints onto durable or shared
storage from there — a periodic rsync, say — is left as a deliberately
separate concern.

```python
from async_checkpoint import AsyncCheckpointer
ckpt = AsyncCheckpointer()
ckpt.save(model, optimizer, step, save_dir, tokenizer=tokenizer)
ckpt.wait_for_pending()   # call before process exit
```

**`local_cache_stream.py`** generalizes a pattern built to survive an
unreliable network on a training box that's otherwise perfectly capable of
running for days unattended. The idea has two halves. The write side reads
from *any* Python generator — not just a specific HF dataset pipeline — and
durably materializes it to a local JSONL file, incrementally, with periodic
flushing, and it stops early and cleanly if the source generator raises
partway through, rather than losing the entire capture to one exception at
row 300,000 of a 500,000-row target. The read side loads that finished
cache into memory once, shuffles it with a given seed, and yields rows in a
loop, reshuffling on every full pass so a long run doesn't see the exact
same row order repeat forever. `train_cpt.py`'s own cache-reading path used
to duplicate this logic inline; it now imports `stream_from_cache` from
here instead.

```python
from local_cache_stream import materialize_to_cache, stream_from_cache
materialize_to_cache(my_generator, "./cache/data.jsonl", target_rows=500_000)
for row in stream_from_cache("./cache/data.jsonl", seed=42):
    ...
```

**`optimizer_compat_guard.py`** is small — one function, really — but it
guards against a failure mode that's genuinely nasty because of *when* it
shows up. Loading a checkpoint's optimizer state into a different optimizer
class than the one that saved it isn't "ignored, harmless." It's been
observed to silently accept the mismatched state and inflate GPU memory
well past what the current optimizer actually needs, and then OOM on the
very first forward pass of the resumed run — which means the failure
doesn't happen at load time, when you'd notice it immediately, but a step
later, minutes into a run you thought had already resumed cleanly. The
realistic way this happens here: you resume without bitsandbytes installed
after training with it, or the reverse (see `bnb_optimizer.py` above).
`check_optimizer_compat()` compares the saved and current optimizer class
names and, on any mismatch, says so and recommends skipping the
optimizer-state load entirely — restart momentum fresh, keep the step
count. Losing Adam's momentum on a switch is a known, bounded cost. Silent
memory corruption is not, and that's the whole reason this exists as its
own guarded decision instead of an assumption baked into the resume path.

```python
from optimizer_compat_guard import check_optimizer_compat
ok, message = check_optimizer_compat(saved_optimizer_type, current_optimizer_type)
```

**`oom_guard.sh`** started as a Mac/Metal script written the day a kernel
panic actually happened: concurrent GPU-memory pressure from two processes
sharing one card corrupted the driver's memory refcounting badly enough to
take the whole kernel down. The fix wasn't clever — poll free memory every
30 seconds, log a warning once it gets tight, and if it crosses a harder
emergency threshold, send SIGTERM to the training process so it dies
*before* the OS or driver reaches an unrecoverable state, not after. That
pattern doesn't care what OS or GPU vendor is underneath it, so this port
swaps the Mac-only `top -l 1` memory parsing for a read of Linux's
`/proc/meminfo` (`MemAvailable`, which already accounts for reclaimable
cache — a better number than raw free memory for deciding whether the
kernel is actually under pressure), which is the realistic target for an
AMD ROCm training server. **What's honestly not in here:** a GPU-VRAM-side
check via `rocm-smi --showmeminfo vram`. That's the natural ROCm-side
equivalent, and the extension point is commented into the script, but this
port was written without access to real ROCm hardware to confirm
`rocm-smi`'s actual output format or how reliably it tracks true
"about-to-OOM" pressure under concurrent load — guessing at that parsing
here would be exactly the kind of unverified claim the rest of this repo
tries to avoid, so it's left as a documented gap instead of a fabricated
implementation. One design note carried over unchanged from the original:
if the process being watched has no SIGTERM handler, this is a hard,
immediate kill, not a clean save, and that's intentional — the goal is to
stop before memory pressure causes real damage, not to guarantee graceful
shutdown after the fact. Pair it with `train_cpt.py`, though, and you get
the graceful case for free: `train_cpt.py` installs its own SIGTERM handler
that checkpoints before exiting, so the two together behave as a real
clean-save-then-exit rather than a hard kill.

```
nohup bash oom_guard.sh <training_pid> > oom_guard.log 2>&1 &
```

## Tips, all from things that actually happened running this on real hardware

- **Reinstall `bitsandbytes` explicitly on every fresh container.** It's easy
  to lose silently on a rebuild, and the failure mode isn't a crash at step 0
  — it's an OOM dozens of iterations in, once the ~4x-larger fallback AdamW
  optimizer state has fully allocated. Confusing to debug if you don't know
  to check for this first.
- **Checkpointing here is local-disk only, no cloud object store.** If you
  need cross-instance durability, sync the checkpoint directory out on your
  own schedule (e.g. a separate rsync loop) rather than assuming any
  in-process cloud upload is wired in — it isn't.
- **`train_cpt.py`'s optional local-JSONL cache mode exists because live
  streaming is only as reliable as your box's network path.** An
  intermittent or blocked connection on a training box is a real, observed
  failure mode. A pre-built local cache trains with zero network dependency
  and just cycles once exhausted.
- **Resuming across a different optimizer type is guarded, not silently
  accepted.** Loading fp32 AdamW state into a bitsandbytes Adam8bit instance
  (or the reverse) inflates memory past what the current optimizer needs and
  OOMs on the first forward pass. `train_cpt.py` checks the saved optimizer's
  class before loading (via `optimizer_compat_guard.py`, above) and skips the
  optimizer state — restarting momentum, keeping the step count — if it
  doesn't match. A bounded, known cost instead of an unbounded, silent one.
- **Batch-size-vs-seqlen tradeoff, measured, not theoretical:** batch=2 at
  seqlen=1024 used *less* memory and stayed stable well past where batch=4
  and batch=2-at-seqlen=2048 both OOM'd at ~99.6% VRAM — attention's
  `O(seqlen²)` scaling means the same total tokens/step can look very
  different depending on how you split batch vs. sequence length. Worth
  testing both directions before assuming one is free.

## Where this hits a real ceiling

Single-GPU throughput is the actual limit here, and it isn't a rounding
error you optimize away — closing an orders-of-magnitude gap to a large
multi-trillion-token CPT target isn't a "just wait longer" problem, it's a
different problem. So it's worth being precise about what single-GPU CPT is
actually good for: targeted or bounded token budgets, domain-adapting a
pruned or expanded model, validating a pipeline end-to-end before scaling
it up. What it isn't: a substitute for real multi-GPU throughput once the
token budget gets large. That's the honest gap — everything here runs
end-to-end on one MI300X today, and the natural next step is the same tools
across several MI300X GPUs instead of one.

## Requirements

Confirmed in active use:

- `torch` (ROCm build for AMD GPUs)
- `safetensors`
- `numpy`
- `transformers` (confirmed working against `5.7.0`; if you're on a
  different version, check whether your `Gemma4Config` registers
  `model_type` as `"gemma4"` or `"gemma4_unified"` — `prune_vocab.py`
  handles that specific mismatch)
- `bitsandbytes` (8-bit Adam; falls back to plain AdamW at ~4x optimizer
  memory if unavailable)

No other versions are pinned in the source this is drawn from — pin what
works in your own environment rather than trusting a fabricated
`requirements.txt`.

## License

Not yet decided — add one before treating this as reusable by others.
