# gemma-mi300x-prune-cpt

Vocabulary pruning + continued-pretraining (CPT) pipeline for a Gemma-4-family
base model, built to run on a **single AMD MI300X GPU** under ROCm/PyTorch.

This repo documents the working, ROCm-native path used to shrink a 12B-parameter
Gemma-4 base checkpoint down to a smaller effective footprint by removing
vocabulary the target use case doesn't need, then running continued
pretraining on the result — entirely on one GPU, no multi-node cluster.

## What's in this repo

- `prune_vocab.py` — Step 1: classifies every tokenizer vocab entry by script
  (CJK / Cyrillic / Arabic / Devanagari / Mongolian / accented Romance-Germanic
  characters) and drops the removable ones, producing a new tokenizer,
  a filtered BPE merge table, and an `_old_to_new_ids.json` remap.
- `prune_embeddings_torch.py` — Step 2: uses that remap to slice the actual
  `embed_tokens.weight` matrix down to the new (smaller) vocab size, rewriting
  only the safetensors shard that contains it and copying the rest of the
  checkpoint through unchanged.
- This README, describing the CPT training setup that runs against the
  pruned checkpoint.

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

Two separate scripts because the ID-remap logic (character classification, BPE
merge filtering, tokenizer.json rewriting) and the tensor-slicing logic (raw
safetensors I/O) are genuinely different concerns, and keeping them separate
means the tokenizer-side logic doesn't need `torch` at all.

## The ROCm/PyTorch training setup

The pruned checkpoint is the input to a continued-pretraining loop
(`windowed_sft_cuda.py` in the source project this repo is drawn from — not
included here in full, since it also touches project-specific data-streaming
code, but the design is documented below because the engineering choices are
the actually-interesting part for a ROCm audience):

- **Optimizer: bitsandbytes 8-bit Adam, with a fallback to `torch.optim.AdamW`
  if bitsandbytes isn't importable.** Both first/second moment buffers are
  kept at ~1 byte/param instead of fp32's 4 bytes/param, which matters a lot
  at this parameter count. This was not a hypothetical concern — on a real run
  against a 14.7B-parameter model, bitsandbytes was missing from a freshly
  rebuilt container, the script's own cross-optimizer safety check correctly
  detected the mismatch and fell back to plain AdamW with a warning (rather
  than silently corrupting state), and the resulting ~4x larger optimizer
  state OOM'd the GPU roughly 110 iterations in, once the optimizer's
  lazily-allocated state had filled in. The fix was reinstalling
  `bitsandbytes`, not a code change — the safety check did exactly its job.
  **Lesson for anyone reproducing this on ROCm: reinstall `bitsandbytes`
  explicitly on every fresh container: it is easy to lose silently and the
  failure mode (OOM dozens of iterations in, not at step 0) is confusing if
  you don't know to look for it.**
- **A cross-optimizer-type resume guard.** Loading an fp32 AdamW checkpoint's
  optimizer state into a freshly-constructed bitsandbytes `Adam8bit` instance
  (or vice versa) is not merely ignored — on a real resume attempt this
  silently accepted the mismatched state and inflated GPU memory to roughly
  2x the expected figure, OOMing on the very first forward pass. The training
  loop now checks the saved optimizer's class name against the current run's
  and skips loading the optimizer state (restarting momentum fresh, but
  keeping the step count) rather than risk that again.
- **Async checkpointing.** Checkpoint writes are split into a synchronous
  phase (copy model + optimizer state from GPU to CPU RAM — this briefly
  blocks training, but a GPU→CPU copy is a small fraction of a full disk
  write) and an asynchronous phase (serialize those CPU tensors to disk on a
  background thread while training continues on the GPU). Bounded to one
  in-flight write at a time to avoid unbounded RAM growth if writes fall
  behind the checkpoint interval. Every checkpoint write (sync or async) goes
  through a write-to-temp-dir-then-atomic-rename pattern, so a `kill -9` or
  `SIGTERM` mid-write can never leave a corrupted checkpoint sitting at the
  path something else will try to load next.
- **Local JSONL cache instead of live HF streaming.** The pipeline supports
  training directly from a category-weighted live Hugging Face stream, but
  also supports pre-materializing that stream into a local JSONL cache and
  training from the cache with **zero network dependency**, cycling it
  indefinitely once exhausted. This exists because live streaming from a
  remote box is only as reliable as that box's network path to the data
  source — an intermittent network on the training box is a real, observed
  failure mode, and a local cache sidesteps it entirely once built.
- **Layer-window freeze/unfreeze**, generalizing "freeze everything outside
  `[start, end)` layers" — lets the same script do either full-model training
  (the default, when 80GB+ of VRAM makes it unnecessary to window) or
  memory-constrained partial-layer training on smaller GPUs.
- **Gradient checkpointing** (recomputes activations in the backward pass
  instead of storing them for every layer) to trade compute time for the
  activation-memory headroom to run a larger batch.

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

Single-GPU throughput is the ceiling here, and it's a real one, not a rounding
error: at the throughput measured on this pipeline (on the order of a few
hundred tokens/sec at batch=1, scaling with batching but still fundamentally
single-GPU), reaching multi-trillion-token CPT targets is not a "wait
longer" problem — it's an orders-of-magnitude gap, the same one every
frontier lab spends thousands of GPUs closing. The honest framing that came
out of measuring this directly: single-GPU CPT is genuinely useful for
targeted, bounded token budgets (adapting a pruned model to a narrower
distribution, running domain-specific continued pretraining, validating a
pipeline end-to-end before scaling it), but it is not a substitute for
multi-GPU throughput once the token budget gets large. That's the gap I'm
looking to close — the pruning + single-GPU CPT pipeline in this repo already
works end-to-end on ROCm, and the natural next step is running the same
pipeline across multiple MI300X GPUs instead of one, which is where the
current setup (single-process, single-device) would need to grow.

## Requirements

Confirmed in active use by this pipeline:

- `torch` (with ROCm build for AMD GPUs)
- `safetensors`
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
versions not pinned beyond what's noted above; pin what works in your own
environment rather than trusting a fabricated `requirements.txt`.

## License

Not yet decided — add one before treating this as reusable by others.
