# gemma-mi300x-prune-cpt

Five real, independently-runnable tools for adapting a Gemma-4-family LLM
checkpoint on a **single AMD MI300X GPU** under ROCm/PyTorch — no multi-node
cluster, no distributed training framework. Built to shrink, grow, and
continue-pretrain a model within the memory budget of one card, and to make
the failure modes of doing that on a single GPU (OOM, crashes, flaky network)
survivable instead of fatal. Every script here is real, run code, not a
from-scratch rewrite for this repo — project-specific naming genericized,
everything else unchanged.

You don't need to use these together or in order — each one solves a
different single-GPU problem on its own:

## `prune_vocab.py` — shrink a tokenizer you don't need in full

Drops vocabulary entries by character-script heuristic (CJK / Cyrillic /
Arabic / Devanagari / Mongolian / accented Latin, configurable) when your use
case doesn't need them, remaps every surviving token to a contiguous new ID
space, filters the BPE merge table to match, and fixes `vocab_size` in both
the top-level and nested config locations some Gemma-4 configs require (missing
one silently reverts the vocab size at load time — a real crash this caught).
Useful on its own any time you want a smaller embedding table without
retraining the tokenizer from scratch.

```
python3 prune_vocab.py --src <base_checkpoint> --dst <pruned_checkpoint>
```

## `prune_embeddings_torch.py` — apply that vocab cut to the actual weights

Takes the ID remap the script above produces and slices
`embed_tokens.weight` down to match, handling both sharded (with an
`index.json`) and single-file checkpoints, rewriting only the shard that
changed and copying the rest through untouched. Useful standalone any time
you've already got a vocab remap and just need the tensor surgery.

```
python3 prune_embeddings_torch.py --src <base_checkpoint> --dst <pruned_checkpoint>
```

## `expand_model.py` — grow a model's width and depth without retraining from scratch

Widens the MLP intermediate dimension and duplicates decoder layers to
increase parameter count from an existing checkpoint. Two different init
strategies depending on what's being added: new width columns get an
**orthogonal-QR init** (real, non-conflicting gradient signal from step one,
not starved like zero-init); newly duplicated layers get **zero-init on
their output projections only** (a true no-op insertion — the layer runs a
real forward pass but contributes nothing to the residual stream until
training turns it on). Also includes an optional GQA fix for full-attention
layers that ship with a single shared KV head and no separate `v_proj` —
worth doing when KV-cache size isn't your actual memory bottleneck.
Pure PyTorch, no Apple-Silicon-only dependency, usable on any checkpoint you
want to grow rather than retrain.

```
python3 expand_model.py --src <pruned_checkpoint> --dst <expanded_checkpoint>
```

**AMD-specific note:** uses `numpy.linalg.qr`, not `torch.linalg.qr`, for the
orthogonal constructions — this ROCm PyTorch build has no LAPACK support for
CPU tensors, so `torch.linalg.qr` on CPU raises directly. Swap it if your
build has working CPU-tensor QR; this repo keeps numpy because it's what
actually ran.

## `train_cpt.py` — continued pretraining, single GPU

The actual CUDA/ROCm training loop: layer-window freeze/unfreeze (full-model
training when VRAM allows, partial-layer windowing when it doesn't), gradient
checkpointing, bitsandbytes 8-bit Adam with a plain-AdamW fallback, async
local-disk checkpointing, a local-JSONL data/cache mode, and clean
SIGTERM-triggered checkpoint-and-exit. Standalone entry point for training
any checkpoint — pruned, expanded, or neither.

```
python3 train_cpt.py --model <checkpoint> --data <jsonl_dir_or_file> --save <out_dir> --batch 1
```

## `catch_and_resume.sh` — keep a single-GPU run alive across crashes

`train_cpt.py` already self-resumes on its own (checks `<save_dir>/training_state.pt`
on startup, no `--resume` flag needed — just re-run the same command). This
wraps that with what self-resume alone doesn't give you: a **loss-tagged
checkpoint history** with rollback if the latest checkpoint's loss spiked
above the best kept one, a **bounded retry** for crashes that keep happening
at the same position (so a real recurring bug doesn't retry silently
forever), and a **stop-file** for a clean shutdown request between attempts.

```
./catch_and_resume.sh
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
  class before loading and skips the optimizer state (restarting momentum,
  keeping the step count) if it doesn't match — a bounded, known cost instead
  of an unbounded, silent one.
- **Batch-size-vs-seqlen tradeoff, measured, not theoretical:** batch=2 at
  seqlen=1024 used *less* memory and stayed stable well past where batch=4
  and batch=2-at-seqlen=2048 both OOM'd at ~99.6% VRAM — attention's
  `O(seqlen²)` scaling means the same total tokens/step can look very
  different depending on how you split batch vs. sequence length. Worth
  testing both directions before assuming one is free.

## Where this hits a real ceiling

Single-GPU throughput is the actual limit, not a rounding error — closing an
orders-of-magnitude gap to a large multi-trillion-token CPT target isn't a
"just wait longer" problem. What single-GPU CPT is genuinely good for:
targeted/bounded token budgets, domain-adapting a pruned or expanded model,
validating a pipeline end-to-end before scaling it. What it isn't: a
substitute for real multi-GPU throughput once the token budget gets large.
That's the actual gap — this all runs end-to-end on one MI300X today, and the
natural next step is the same tools across several MI300X GPUs instead of one.

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
