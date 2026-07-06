# gemma-mi300x-prune-cpt

A vocabulary-pruning → model-expansion → continued-pretraining (CPT) pipeline
for a Gemma-4-family base model, built and actually run end-to-end on a
**single AMD MI300X GPU** under ROCm/PyTorch. No multi-node cluster, no
distributed training framework — one GPU, real memory constraints, and the
engineering that comes out of working within them.

This repo is the working, ROCm-native path used to take a 12B-parameter
Gemma-4 base checkpoint, prune vocabulary the target use case doesn't need,
expand the model's width and depth to grow its parameter count, and then run
continued pretraining on the result. Every script here is real, run code —
not a from-scratch rewrite for this repo — with any project-specific naming
genericized. See `NEXT_STEPS_FOR_SEAN.md` for exactly what was changed and
why.

## Pipeline order

```
prune_vocab.py            (tokenizer surgery: drop unneeded vocab, remap IDs)
        |
        v
prune_embeddings_torch.py (tensor surgery: slice embed_tokens to the new vocab)
        |
        v
expand_model.py            (width + depth expansion: grow the pruned model)
        |
        v
train_cpt.py                (continued pretraining, supervised by catch_and_resume.sh)
```

- **`prune_vocab.py`** — Step 1: classifies every tokenizer vocab entry by
  script (CJK / Cyrillic / Arabic / Devanagari / Mongolian / accented
  Romance-Germanic characters) and drops the removable ones, producing a new
  tokenizer, a filtered BPE merge table, and an `_old_to_new_ids.json` remap.
- **`prune_embeddings_torch.py`** — Step 2: uses that remap to slice the
  actual `embed_tokens.weight` matrix down to the new (smaller) vocab size,
  rewriting only the safetensors shard that contains it and copying the rest
  of the checkpoint through unchanged.
- **`expand_model.py`** — Step 3: widens the MLP intermediate dimension and
  adds new decoder layers, growing the pruned checkpoint's parameter count
  (the source pipeline used this to go from a pruned 12B base up to a
  ~15B-parameter model). PyTorch-native, no Apple-Silicon-only dependency.
- **`train_cpt.py`** — Step 4: the actual CUDA/ROCm continued-pretraining
  loop that trains the expanded checkpoint on a raw-text or chat-formatted
  dataset.
- **`catch_and_resume.sh`** — a crash-recovery supervisor that wraps
  `train_cpt.py`, relaunching it automatically after a crash and keeping a
  loss-tagged checkpoint history so a bad training patch can be rolled back.

Four separate scripts (plus the supervisor) because each stage is a
genuinely different concern: tokenizer-side ID remapping doesn't need
`torch` at all, tensor slicing is raw safetensors I/O, model expansion is a
one-shot offline transform, and training is the actual GPU-bound loop.

## The pruning pipeline, in order

1. **`prune_vocab.py --src <base> --dst <pruned>`** reads `tokenizer.json`,
   classifies every vocab token by a **character-range heuristic** (not real
   language ID — a token is only dropped if it *contains* a character from a
   targeted script/diacritic set; plain-Latin words from those languages that
   happen to overlap with common English subwords are untouched). Special/added
   tokens (`<pad>`, `<bos>`, modality placeholders, etc.) are always kept
   regardless of script. It then:
   - Rebuilds the vocab with contiguous new IDs (`0..N-1`), preserving original
     relative ordering.
   - Filters the BPE merge table so a merge `(a, b) -> ab` only survives if `a`,
     `b`, and the merged result all survived pruning.
   - Remaps every `added_tokens` entry's ID into the new ID space (hard-fails
     if a special token was accidentally dropped — this should never happen,
     since special tokens are protected earlier in the same pass).
   - Updates `vocab_size` in **both** the top-level `config.json` field and the
     nested `text_config.vocab_size` — some Gemma-4 model-arg classes overwrite
     the nested field from the top-level one at load time, so setting only one
     of them can silently revert to the old vocab size at model-build time
     (this exact failure mode showed up as a strict-shape-mismatch crash before
     the fix).
   - Writes `_old_to_new_ids.json`, the ID remap the next script needs.
2. **`prune_embeddings_torch.py --src <base> --dst <pruned>`** reads that remap
   and does the actual tensor surgery: it locates whichever safetensors shard
   holds `model.language_model.embed_tokens.weight` (handling both a sharded
   multi-file checkpoint with an `index.json` and a single unsharded
   `model.safetensors` file with no index — it synthesizes one in that case),
   slices the embedding matrix down to just the kept rows, and rewrites only
   that shard. Every other shard is copied through byte-for-byte. It regenerates
   `model.safetensors.index.json` with the corrected total size and parameter
   count delta.

## Expansion: `expand_model.py`

Grows the pruned checkpoint's width (MLP intermediate dimension) and depth
(number of decoder layers) rather than training a new, larger model from
scratch. Two deliberately different initialization strategies for the two
kinds of new capacity:

- **New width (new MLP columns within existing layers): orthogonal-QR init**,
  scaled down. Built via `numpy.linalg.qr` on a random matrix so the new
  columns start close to orthogonal to what the model already learned — not
  conflicting with existing representations, but with a real (non-zero)
  gradient signal from step one, unlike a pure zero-init which would starve
  those specific new columns of any initial gradient.
- **New depth (newly duplicated layers): zero-init on the output
  projections only.** A duplicated layer clones its donor's real, trained
  weights, but its `self_attn.o_proj` and `mlp.down_proj` are zeroed — the
  layer computes a genuine internal forward pass but contributes nothing to
  the residual stream at insertion time (a true identity/no-op), and training
  gradually turns that contribution on.

Also includes an optional GQA fix (`--gqa-kv-heads`, default 8): some
full-attention layers in this model family ship with an extreme MQA setup —
a single shared KV head with V literally reusing K, confirmed by inspecting
the real checkpoint's safetensors header (no separate `v_proj` key exists for
those layers at all). On a GPU where KV-cache size at your target context
isn't actually the memory bottleneck, that's a real quality cost for a memory
saving you don't need. This grows `k_proj` via orthogonal padding (preserving
already-learned K directions) and builds a fresh `v_proj` from scratch via
orthogonal QR init.

**Numpy QR, not `torch.nn.init.orthogonal_`, for all of the orthogonal
constructions above** — the ROCm PyTorch build this pipeline ran on doesn't
have LAPACK support for CPU tensors (`torch.linalg.qr` on a CPU tensor raises
a direct error asking for a LAPACK-enabled build), while numpy's own LAPACK
bindings work. If your build has working CPU-tensor `torch.linalg.qr`, you
can swap it — this repo keeps numpy because it's the version that's actually
been run against real hardware.

Output is written as standard sharded safetensors with a real
`model.safetensors.index.json`, splitting at a configurable byte budget
(default 5GB/shard) — loadable the same way any HF sharded checkpoint is.

## Training: `train_cpt.py`

The pruned-then-expanded checkpoint is the input to a continued-pretraining
loop. Four engineering points worth calling out specifically, all grounded in
what actually happened running this pipeline (not aspirational design):

### 1. bitsandbytes 8-bit Adam, and a real OOM it caused when missing

Both optimizer moment buffers are kept at ~1 byte/param instead of fp32's 4
bytes/param, which matters a lot at this parameter count. The script falls
back to plain `torch.optim.AdamW` if `bitsandbytes` isn't importable — and
that fallback path is not a hypothetical concern. On a real run against a
14.7B-parameter model, `bitsandbytes` was missing from a freshly rebuilt
container. The training script's own cross-optimizer safety check (see #4
below) correctly detected the mismatch and fell back to plain AdamW with a
clear warning, exactly as designed — but the resulting ~4x larger optimizer
state OOM'd the GPU roughly 110 iterations in, once the optimizer's
lazily-allocated state had fully filled in. The fix was reinstalling
`bitsandbytes`, not a code change.

**HARD RULE if you're reproducing this: reinstall `bitsandbytes` explicitly
on every fresh container.** It's easy to lose silently, and the failure mode
(OOM dozens of iterations into a run, not at step 0) is genuinely confusing
if you don't know to look for it — you'll be debugging a "random" OOM instead
of a one-line reinstall.

### 2. Async checkpointing, local disk only — no cloud object store

Checkpoint writes are split into a synchronous phase (copy model + optimizer
state from GPU to CPU RAM — this briefly blocks training, but a GPU→CPU copy
is a small fraction of a full disk write) and an asynchronous phase (serialize
those CPU tensors to disk on a background thread while training continues on
the GPU). Bounded to one in-flight write at a time to avoid unbounded RAM
growth if writes fall behind the checkpoint interval. Every write, sync or
async, goes through a write-to-temp-dir-then-atomic-rename pattern, so a
`kill -9` or `SIGTERM` mid-write can never leave a corrupted checkpoint
sitting at the path something else will try to load next.

**This design writes to local disk only — it does not push to any cloud
object store (no GCS, no S3, nothing).** An earlier draft of this training
script's own docstring described a GCS-checkpointing contract (periodic
`gsutil cp -r` up/down for resuming across preemptible-instance restarts).
That was accurate for a different deployment context this script was
originally written for, but it is **not** how this pipeline actually runs on
the single-GPU box described in this repo: there, training checkpoints
locally, and a separate background process periodically rsyncs the
checkpoint directory to network storage on its own schedule, decoupled from
the training loop entirely. `train_cpt.py` in this repo reflects that real
design — local atomic writes, no in-process cloud dependency — rather than
the docstring aspiration. If you want cloud-backed checkpointing, that's a
separate concern to add on top, not something to assume is already wired in.

### 3. Local JSONL cache instead of live streaming

The training loop can read a packed CPT dataset from a local JSONL file or
directory (`--data`), or from a local JSONL **cache** (`--cpt-cache`) that's
been pre-materialized ahead of time from a public dataset source. The cache
path exists because live streaming from a remote data source is only as
reliable as the training box's network path to it — an intermittent or
outright blocked network connection on the training box is a real, observed
failure mode, not a theoretical one, and a local cache sidesteps it entirely
once built: `--cpt-cache` trains with **zero network dependency**, cycling
the cache indefinitely once exhausted (better than stopping, since there's no
network to refill it). This repo's `train_cpt.py` includes the cache-reading
path; the separate cache-building/materialization step (pulling from a live
source with its own per-source timeout/retry handling) is not included here,
since that piece is tied to project-specific dataset source configuration.

### 4. A cross-optimizer-type resume guard

Loading an fp32 AdamW checkpoint's optimizer state into a freshly-constructed
`bitsandbytes` `Adam8bit` instance (or vice versa) is not "ignored,
harmless." On a real resume attempt, doing this silently accepted the
mismatched state and inflated GPU memory well beyond what the current
optimizer should need, OOMing on the very first forward pass. `train_cpt.py`
checks the saved optimizer's class name against the current run's before
loading anything, and — if they differ — **skips loading the optimizer
state entirely**, restarting that optimizer's momentum fresh while still
resuming the step count. Losing optimizer momentum on an optimizer-type
switch is a known, bounded cost. Silently corrupting memory usage is not,
and this is the design decision that refuses to trade one for the other.

### Also in `train_cpt.py`

- **Layer-window freeze/unfreeze**, generalizing "freeze everything outside
  `[start, end)` layers" — lets the same script do either full-model training
  (the default, when 80GB+ of VRAM makes windowing unnecessary) or
  memory-constrained partial-layer training on smaller GPUs.
- **Gradient checkpointing** (recomputes activations in the backward pass
  instead of storing them for every layer) to trade compute time for the
  activation-memory headroom to run a larger batch.
- **SIGTERM handling** — on receiving SIGTERM, the training loop finishes its
  current step, writes a checkpoint, and exits cleanly, so a preemptible or
  time-limited instance doesn't lose an in-progress step's work.

## Crash recovery: `catch_and_resume.sh`

`train_cpt.py` already self-resumes: on startup it checks whether
`<save_dir>/training_state.pt` exists, and if so, loads model + optimizer +
step count from it automatically. There's no `--resume` or `--start-iter`
flag to pass — re-running the identical command is the entire resume
mechanism.

`catch_and_resume.sh` is a bash supervisor that relaunches `train_cpt.py`
automatically after a crash, adding what the script's built-in resume doesn't
cover on its own:

- **Loss-tagged checkpoint history.** `train_cpt.py --save` is a single slot
  that gets atomically overwritten every checkpoint interval. Without a side
  history, a checkpoint written during a bad patch of training (e.g. a data
  or LR issue that spikes the loss) permanently replaces the last known-good
  state with no way back. The supervisor keeps the last few loss-tagged
  copies and can roll back to the best one if the current state's loss is
  too far above it.
- **Retry with a same-position stall detector.** If a relaunch crashes
  without making it past the last checkpoint, the supervisor retries (assumed
  transient) up to a bounded number of times before giving up and treating it
  as a real, recurring problem instead of retrying forever silently.
- **A stop-file** so you can request a clean stop between attempts without
  killing the process mid-write.

This pattern (loss-tagged history, rollback-on-regression, retry-with-backoff,
stop-file) is adapted from a supervisor originally written for a different,
MLX/Metal-based training script on Apple Silicon — that script needed
explicit `--resume`/`--start-iter` flags on every relaunch because its
checkpoint filenames used a step counter that reset on every invocation.
`catch_and_resume.sh` in this repo is a fresh script, not a copy of that one,
written against `train_cpt.py`'s actual (and different) self-resume contract
— see the comment block at the top of the script for the specifics.

### A real, measured batching lesson

Two OOM crashes at larger batch sizes (batch=4, and batch=2 at seqlen=2048)
both died at roughly 99.6% of the GPU's memory. Since attention compute
scales roughly O(seq_len²) per sequence, batch=2 at seqlen=1024 uses *less*
memory than batch=1 at seqlen=2048 for the same total tokens per step
(2×1024² < 1×2048²) — switching to the smaller-seqlen/larger-batch
configuration has been stable well past the iteration count where the other
configurations OOM'd. This is a concrete, measured example of how attention's
quadratic scaling interacts with batch-vs-seqlen tradeoffs on real hardware,
not a general claim — your numbers will differ by model size, sequence
length, and how much of the model is actually unfrozen.

## Where I'm hitting limits

Single-GPU throughput is the ceiling here, and it's a real one, not a
rounding error: at the throughput measured on this pipeline (on the order of
a few hundred tokens/sec at batch=1, scaling with batching but still
fundamentally single-GPU), reaching multi-trillion-token CPT targets isn't a
"wait longer" problem — it's an orders-of-magnitude gap, the same one every
larger training effort spends many more GPUs closing. The honest framing
that came out of measuring this directly: single-GPU CPT is genuinely useful
for targeted, bounded token budgets (adapting a pruned/expanded model to a
narrower distribution, running domain-specific continued pretraining,
validating a pipeline end-to-end before scaling it), but it isn't a
substitute for multi-GPU throughput once the token budget gets large. That's
the gap I'm looking to close — the prune → expand → CPT pipeline in this
repo already works end-to-end on ROCm, on one GPU, and the natural next step
is running the same pipeline across multiple MI300X GPUs instead of one,
which is where the current setup (single-process, single-device) would need
to grow.

## Requirements

Confirmed in active use by this pipeline:

- `torch` (with ROCm build for AMD GPUs)
- `safetensors`
- `numpy`
- `transformers` (a Gemma-4-family checkpoint used with this pipeline was
  loaded successfully against transformers `5.7.0` — that's the one pinned
  version confirmed by direct observation; if you're on a different version,
  check whether your installed `Gemma4Config` registers `model_type` as
  `"gemma4"` or `"gemma4_unified"`, since `prune_vocab.py` handles that
  specific mismatch)
- `bitsandbytes` (for 8-bit Adam — see the reinstall note above; falls back
  to plain `torch.optim.AdamW` if unavailable, at roughly 4x the optimizer
  memory cost)

No other package versions are pinned in the source this repo is drawn from —
pin what works in your own environment rather than trusting a fabricated
`requirements.txt`.

## License

Not yet decided — add one before treating this as reusable by others.
