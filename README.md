# gemma-mi300x-prune-cpt

Fourteen real, independently-runnable tools (twelve Python + two shell) for
adapting an LLM checkpoint on a **single AMD GPU** under ROCm/PyTorch — no
multi-node cluster, no distributed training framework. They came out of
actually doing this once, for real: shrinking a tokenizer, growing a model,
and continue-pretraining it, all on one MI300X, and then hitting the
specific ways a single GPU fails you (OOM, crashes, a data source that goes
unreachable mid-run) and fixing each one for real instead of writing around
it. Every script here is real, run code, not a from-scratch rewrite for this
repo — core training logic unchanged; refactored into standalone modules
and parameterized into CLI flags.

**A note before you go further, since it's the kind of thing worth saying up
front instead of burying:** this repo's own real runs all use full-parameter
fine-tuning (`train_cpt.py`), not LoRA. There's a `lora_train.py` here too,
and it's a real, working script — but it's offered as an option for people
who specifically want LoRA's memory/storage tradeoffs, not because it's what
this repo recommends. If you skip straight to the tools list below, that
framing is repeated there in the same terms, not softened.

None of this is pinned to one GPU. There's no device-name check, no
architecture branch, no hardcoded VRAM figure anywhere in the source —
batch size, sequence length, and how many layers stay unfrozen are all
plain CLI flags, so the same scripts scale down to a smaller card by
freezing more layers and shrinking the batch, or scale up by unfreezing
more and running bigger batches. Standard ROCm/PyTorch throughout — nothing
here calls out to an MI300X-only code path. It happens to have been built
and run on one MI300X; there's nothing in it that ties it there.

Model-family lock-in got the same treatment, and it's worth being precise
about where that claim actually stands rather than rounding it up. Every
model-specific constant that used to be hardcoded — the embedding tensor's
key name, the vocab_size config path, the layer-naming prefix, the sharding
size, the depth/width step sizes, the GQA head count — is a CLI flag now,
defaulting to the Gemma-4 layout these were built against but pointable at
whatever your own checkpoint actually uses. The one piece that doesn't
become a flag is `expand_model.py`'s submodule key *suffixes*
(`gate_proj`/`up_proj`/`down_proj`, `q_proj`/`k_proj`/`v_proj`/`o_proj`) —
and having actually gone and checked the installed `transformers` library's
own modeling source rather than assuming, those suffixes turn out to be
shared by most Llama-derived decoder architectures (Llama, Mistral,
Qwen2/3, every Gemma generation), not just Gemma's. So "Gemma-4-specific"
undersold it — but it's still not universal: GPT-2, the original Phi, Phi-3,
Falcon, MPT, and BLOOM all use genuinely different, fused-QKV naming and
would need real code changes, not a flag, to support. The per-tool sections
below spell out exactly what's been checked where, including the one place
that stayed narrowly Gemma-4-specific on purpose: `expand_model.py`'s GQA
fix now runs an actual detection pass against the loaded checkpoint's
tensors before touching anything, and skips cleanly with a warning if the
checkpoint doesn't match the layout it targets, rather than assuming every
input does.

You don't need to use these together or in order — each one solves a
different single-GPU problem on its own. The canonical pipeline order, if
you use them together, is:

```
prune_vocab.py → prune_embeddings_torch.py → expand_model.py → [mtp_head.py] → train_cpt.py / train_sft.py
```

Skip whichever steps you don't need — pruning, expansion, and MTP are all
independently optional, and training directly against an unmodified base
checkpoint is a completely normal way to use `train_cpt.py` (or
`train_sft.py`, its SFT-only alias — see below) on its own. `lora_train.py`
can substitute for that last step too, if you specifically want LoRA
instead of full-parameter training; it's covered in its own section further
down, framed the same way it should be: an option, not the recommendation.

## Table of Contents

- [Installation](#installation)
- [Tools](#prune_vocabpy--shrink-a-tokenizer-you-dont-need-in-full)
  - [`prune_vocab.py`](#prune_vocabpy--shrink-a-tokenizer-you-dont-need-in-full)
  - [`prune_embeddings_torch.py`](#prune_embeddings_torchpy--apply-that-vocab-cut-to-the-actual-weights)
  - [`expand_model.py`](#expand_modelpy--grow-a-models-width-and-depth-without-retraining-from-scratch)
  - [`mtp_head.py`](#mtp_headpy--add-a-multi-token-prediction-head)
  - [`train_cpt.py`](#train_cptpy--continued-pretraining-single-gpu)
  - [`train_sft.py`](#train_sftpy--the-sft-only-name-for-train_cptpy)
  - [`lora_train.py`](#lora_trainpy--lora-via-peft-offered-not-recommended)
  - [`catch_and_resume.sh`](#catch_and_resumesh--keep-a-single-gpu-run-alive-across-crashes)
  - [Standalone utilities](#standalone-utilities)
  - [`rocm_env.py`](#rocm_envpy--amd-gpu-arch-detection--override)
- [Tips / Troubleshooting](#tips--troubleshooting)
- [Where this hits a real ceiling](#where-this-hits-a-real-ceiling)
- [Testing](#testing)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

**Option 1: Docker (recommended — bakes in all deps including ROCm torch):**

```bash
docker build -t gemma-prune-cpt .
docker run --device /dev/kfd --device /dev/dri --group-add video \
           --shm-size 64G -v $(pwd):/work -w /work -it gemma-prune-cpt \
           python3 train_cpt.py --model ... --save ...
```

**Option 2: pip (install ROCm torch first from [pytorch.org](https://pytorch.org/get-started/locally/), then the rest):**

```bash
pip install -r requirements.txt
```

`requirements.txt` lists tested-known-good versions. `torch` is NOT in it —
the ROCm build must come from AMD's index, not PyPI. Install it first.

**AMD consumer/older cards:** if your GPU's gfx arch isn't in the torch
wheel's compiled list (common on RDNA1/2, older cards), `train_cpt.py` calls
`rocm_env.py` automatically at startup to detect and set
`HSA_OVERRIDE_GFX_VERSION`. See [`rocm_env.py`](#rocm_envpy--amd-gpu-arch-detection--override)
below. You can also force it with `--gfx-override gfx1100`.

## `prune_vocab.py` — shrink a tokenizer you don't need in full

A Gemma-4 tokenizer ships vocabulary for scripts you may never see — CJK,
Cyrillic, Arabic, Devanagari, Mongolian, a long tail of accented Latin. If
your use case doesn't need all of that, `prune_vocab.py` drops those entries
by a character-script heuristic, remaps every surviving token to a
contiguous new ID space, and filters the BPE merge table to match. That
part — the classification and the vocab/merge surgery — only ever looks at
token strings and `tokenizer.json`; there's nothing Gemma-specific in it, so
it runs the same way against any tokenizer.

The `config.json` side is where the model-family specifics live, and
they're flag-driven rather than hardcoded. One detail that mattered in
practice: some Gemma-4 configs store `vocab_size` in two places, a top-level
field and a nested one, and missing either one silently reverts the vocab
size at load time — this script fixes both by default (`--vocab-size-paths`
if your own config nests it somewhere else), and it fixes both because a
real load crashed on exactly this the first time around. A second, narrower
fix renames `model_type: "gemma4_unified"` to `"gemma4"` when a checkpoint's
`config.json` uses the older string but the installed `transformers`
registers Gemma4Config under the shorter name — it only fires on that exact
match, so it's already a no-op against any other model family's config.
Useful on its own any time you want a smaller embedding table without
retraining the tokenizer from scratch.

```
python3 prune_vocab.py --src <base_checkpoint> --dst <pruned_checkpoint>
```

## `prune_embeddings_torch.py` — apply that vocab cut to the actual weights

Dropping tokenizer entries doesn't shrink anything until the model's actual
weights follow. This script takes the ID remap the tool above produces and
slices the embedding tensor down to match, handling both sharded (with an
`index.json`) and single-file checkpoints — it rewrites only the shard that
changed and copies the rest through untouched, rather than reserializing
weights it didn't need to touch. Which tensor gets sliced is a `--embed-key`
flag (defaults to the Gemma-4-family key,
`model.language_model.embed_tokens.weight`); everything past that lookup —
reading one named tensor out of the state dict, slicing its rows, writing
it back — doesn't care what architecture it came from. Point `--embed-key`
at whatever your own checkpoint's safetensors header actually calls its
embedding weight (e.g. plain `model.embed_tokens.weight` on many non-Gemma
architectures) and the rest just works — though that's only been exercised
against the Gemma-4-family key by this repo, not verified against another
architecture. Useful standalone any time you've already got a vocab remap
and just need the tensor surgery.

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
training turns it on.

There's also an optional GQA fix for full-attention layers that ship with a
single shared KV head and no separate `v_proj` at all — worth applying when
KV-cache size isn't your actual memory bottleneck, since the compression is
otherwise just trading away model quality for a saving you don't need. This
used to be a fix that quietly assumed every checkpoint it touched had that
exact layout. It doesn't anymore: before rewriting anything, it now checks
the loaded checkpoint's real tensors — does a `v_proj` key actually exist
for these layers (it shouldn't, if the fix applies), does `k_proj`'s real
shape agree with what the config claims the kv-head count is — and skips
the pass cleanly with a specific reason logged if either check fails,
instead of running anyway and either crashing on a shape mismatch or
silently overwriting a real, already-trained V projection on an
architecture that has one. `--force-gqa-fix` is there for the rare case
you've verified by hand that the fix is still correct despite a failed
check; don't reach for it unless you actually have.

Uses PyTorch + numpy + safetensors (no Apple-Silicon-only MLX dependency).
The width/depth expansion logic and the tensor-key *prefix* are both
parameterized and have been checked against how mainstream decoder
architectures actually name their weights (see the intro above and the
module docstring for the verified list of what matches and what doesn't) —
but this has only ever been run end-to-end against Gemma-4-family
checkpoints on one MI300X. Retargeting a genuinely different naming
convention (fused-QKV architectures especially) still needs the submodule
key suffix edits noted in the docstring, not just a flag.

```
python3 expand_model.py --src <pruned_checkpoint> --dst <expanded_checkpoint>
```

**AMD-specific note:** it uses `numpy.linalg.qr`, not `torch.linalg.qr`, for
the orthogonal constructions — this ROCm PyTorch build has no LAPACK support
for CPU tensors, so `torch.linalg.qr` on CPU raises directly, not
approximately, not sometimes. Swap it if your build has working CPU-tensor
QR; this repo keeps numpy because it's what actually ran.

## `mtp_head.py` — add a Multi-Token-Prediction head

Standalone tool that appends real MTP modules to an expanded checkpoint,
following the **DeepSeek-V3 MTP pattern**: per depth, an RMSNorm + a
`2*hidden → hidden` projection (orthogonally initialized) + one transformer
block **cloned from the last real decoder layer** (real pretrained weights,
not fresh init) + a final RMSNorm. The weights are written as a safetensors
shard and merged into the checkpoint's index, and `config.json` is updated
with `mtp_depths` / `mtp_loss_weight` / `auto_map`.

This replaces an earlier version of `expand_model.py` whose docstring claimed
to instantiate MTP modules and append them as a shard — that claim was false
(the code only wrote two config fields). `mtp_head.py` is the real
implementation; `expand_model.py` no longer touches MTP at all.

**What it does NOT provide:** the modeling Python code. For the generated
weights to be used at train/inference time, you need a `modeling_custom.py`
(alongside the checkpoint) defining a `CustomForCausalLM` class whose forward
instantiates MTP modules consuming the keys `mtp_head.py` documents
(`model.mtp_layers.{i}.enorm.weight`, `.eh_proj.weight`, `.block.<suffix>`,
`.lnorm.weight`, `model.mtp.norm.weight`). `mtp_head.py` produces correct
weights + config; the modeling code is your responsibility.

```
python3 mtp_head.py --src <expanded_checkpoint> --dst <mtp_checkpoint>
```

## `train_cpt.py` — continued pretraining, single GPU

This is the actual CUDA/ROCm training loop, and the rest of this README is
mostly about the problems it ran into and how they got fixed: layer-window
freeze/unfreeze (full-model training when VRAM allows, partial-layer
windowing when it doesn't), gradient checkpointing, an 8-bit-Adam-with-AdamW-
fallback optimizer, async local-disk checkpointing (opt-in via
`--async-checkpoint`, off by default), a local-JSONL data/cache
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

### AMD-specific optimizations (all opt-in, throughput numbers not yet verified)

`train_cpt.py` supports several AMD-ROCm-specific optimizations, each behind a
CLI flag. The optimization flags (--flash-attn, --dtype fp8, --compile) fall
back gracefully to the default path (bf16, eager, standard attention) if their
dependency isn't installed; --ddp and --profile are infrastructure flags
without a fallback (they either run or don't, based on whether you pass them).
**Honest caveat up front:** the code paths below are real and exercised by
this repo's own logic (the fallback branches, the DDP rank/all-reduce
plumbing, the flag wiring), but none of the throughput/speedup figures
mentioned (e.g. "~2x", "2-4x") have been measured against real ROCm hardware
by this repo — they're the figures commonly cited for these techniques in
general, not something benchmarked here. Configurable is not the same claim
as verified; treat every number below as "expected, unconfirmed on this
codebase's own hardware" until you've run it yourself:

- **`--flash-attn`** — Flash Attention 2. Reduces attention VRAM from
  `O(seqlen²)` to `O(seqlen)`, which is the mechanism that speeds up
  long-context training in general — directly attacks the OOM theme this repo
  is built around. Requires `flash-attn` built for ROCm (`pip install
  flash-attn --no-build-isolation`). Falls back to standard attention with a
  warning if not installed.
- **`--dtype fp8`** — fp8 training via `torchao`'s `Float8Linear`
  (`float8_e4m3fn`). MI300X/MI325X have native fp8 compute, which is why fp8
  is expected to be faster than bf16 on those cards — the actual multiplier
  hasn't been measured here. Falls back to bf16 if `torchao` isn't installed
  or the card lacks fp8 hardware.
- **`--compile`** — `torch.compile()` with ROCm's inductor backend for kernel
  fusion + graph optimization. First few steps are slower (compilation), then
  faster. Falls back to eager mode if compilation fails.
- **`--profile <dir>`** — `torch.profiler` trace (viewable in
  `chrome://tracing` or Perfetto) including ROCm/HIP kernel launches. For
  kernel-level profiling beyond torch.profiler, wrap the run with
  `rocprof --stats python3 train_cpt.py ...`.
- **`--hip-alloc-conf`** — sets `PYTORCH_HIP_ALLOC_CONF` (default
  `max_split_size_mb:128`) to prevent the caching allocator fragmentation that
  causes phantom OOMs on long runs. Handled by `rocm_env.py` alongside the gfx
  override.
- **`--ddp`** — multi-GPU training via `torch.distributed` +
  `DistributedDataParallel`. Launch with `torchrun --nproc_per_node=N
  train_cpt.py --ddp ...`. Only rank 0 writes checkpoints/logs; all ranks
  participate in gradient all-reduce. The rank/device/all-reduce wiring is
  real code, but this repo has only ever run on a single GPU — the multi-GPU
  path itself (not just its speedup) is untested against real multi-GPU
  hardware. Verify it actually converges correctly on your own cluster before
  trusting it for a real run.

These flags are designed to compose (`--ddp --flash-attn --dtype fp8
--compile` on a multi-GPU MI300X box), but that combination specifically has
not been run end-to-end here either.

## `train_sft.py` — the SFT-only name for `train_cpt.py`

Here's a thing worth being direct about: `train_cpt.py` already does SFT.
It's the default mode — omit `--cpt` and you get chat-template tokenization
with assistant-turn-only loss masking (`build_sft_example()`, described
above). So why does a second file exist? Purely so that scanning this
repo's file list — `ls`, a GitHub file browser, whatever — actually shows
both a CPT tool and an SFT tool without making someone open `train_cpt.py`
and read its docstring to learn the two are the same file behind a flag.
That's the entire job of `train_sft.py`: a thin wrapper that rewrites its
own argv into the equivalent `train_cpt.py` invocation and calls
`train_cpt.main()` directly — no second training loop, no second copy of
`build_sft_example()`, nothing that could drift out of sync with the real
implementation. It also refuses `--cpt` outright with a clear error rather
than silently accepting it, since accepting it would defeat the point of a
separately-named SFT entry point. Every other `train_cpt.py` flag —
checkpointing, DDP, flash-attn, compile, fp8, gfx override — passes through
unchanged.

```
python3 train_sft.py --model <checkpoint> --data <jsonl_dir_or_file> --save <out_dir> --batch 1
```

`train_sft.py --selftest` runs its own argv-rewrite/refusal logic first,
then delegates straight to `train_cpt.py`'s real `self_test()` — so the
actual SFT logic is tested exactly once, in exactly one place, same as it's
implemented.

## `lora_train.py` — LoRA via `peft`, offered, not recommended

Say this first, because it's the part that matters most if you're skimming:
**this repo's own real runs all use full-parameter fine-tuning
(`train_cpt.py` / `train_sft.py`), and that's still the recommended path
here.** `lora_train.py` exists because some developers specifically want
LoRA for its memory and storage tradeoffs — a much smaller optimizer
footprint, adapter checkpoints in the tens of MB instead of tens of GB —
not because it's a better default. The honest tradeoff, stated plainly in
both directions: LoRA needs less VRAM and produces much smaller checkpoints,
but only updates a small low-rank slice of the weight space, which
generally adapts a model less than full fine-tuning does for the same data
and step count. Full-parameter training updates every weight and is what
this repo's own engineering narrative — the OOM war stories, the
batch/seqlen tradeoffs, everything in the Tips section — is actually based
on. If your GPU can afford full-parameter training (this repo's whole
premise is that a single 80GB+ AMD GPU usually can), that's still the
better default; reach for `lora_train.py` when you specifically want the
memory/storage win badly enough to accept weaker adaptation for it.

Built on `peft.LoraConfig` / `peft.get_peft_model` — HuggingFace's own,
widely-tested LoRA implementation — rather than hand-rolled low-rank math,
because reimplementing that math here would just be a second, less-tested
copy of something `peft` already gets right. `--target-modules` defaults to
the same q/k/v/o_proj + gate/up/down_proj submodule names the rest of this
repo already targets, checked the same way `expand_model.py`'s tensor-suffix
generalization was checked — against the installed `transformers` library's
own modeling source, confirmed real for Llama, Mistral, Qwen2, Qwen3, and
every Gemma generation (see `expand_model.py`'s docstring for the full
verified exceptions list; point `--target-modules` at your own architecture's
real submodule names if it's one of those exceptions). Reuses
`train_cpt.py`'s data loading, tokenization, masking, collation, and eval
functions directly (`load_jsonl`, `build_sft_example` / `build_cpt_example`,
`collate`, `pack_examples`, `run_eval`, `lr_at_step`) rather than
duplicating them — same "no duplicated logic" rule as everywhere else in
this repo. What it does NOT reuse, because it doesn't apply to LoRA: the
layer-window freeze/unfreeze logic (`peft` already restricts trainable
parameters to the adapter, so there's no separate freeze-window concept),
the atomic full-checkpoint rename dance (adapter saves are small enough
that `peft`'s own `save_pretrained` is a reasonable checkpoint unit on its
own), and DDP/fp8/torch.compile (out of scope for what's meant to be a
lighter-weight path — full-scale multi-GPU training is `train_cpt.py --ddp`).

```
python3 lora_train.py --model <checkpoint> --data <jsonl_dir_or_file> --save <out_dir> \
    --iters 2000 --lora-r 16 --lora-alpha 32

# Also merge the adapter into the base weights and save a full standalone
# checkpoint (bigger on disk, loadable with plain
# AutoModelForCausalLM.from_pretrained(), no peft import required):
python3 lora_train.py --model <checkpoint> --data <jsonl_dir_or_file> --save <out_dir> --merge-and-save
```

`lora_train.py --selftest` builds a tiny *real* Llama-architecture model via
`transformers` (Llama is one of the confirmed-matching architectures for
the default target modules), wraps it with a real `peft.LoraConfig`, and
runs an actual forward pass, backward pass, adapter save, adapter reload
onto a fresh base model, and `merge_and_unload()` — checking at each step
that gradients land only on LoRA parameters, that the reloaded adapter
weights match exactly what was saved, and that the merged model has zero
`lora_` keys left. No GPU required; needs `peft` installed (see
`requirements.txt`).

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

Four of these came directly out of `train_cpt.py`'s `main()` — pieces that
were doing real, non-trivial work but only existed as prose and inline logic
buried there, worth pulling out on their own merits (optimizer construction,
async checkpoint writes, the optimizer-type resume guard, and local-cache
data streaming). A fifth is a port of a memory-safety script that started
life solving a Mac-specific crash but whose actual pattern — poll, warn,
kill before the OS does something worse — has nothing Mac-specific about it.

**`bnb_optimizer.py`** exists because "which optimizer did this run
actually get" turns out to matter a lot on a single GPU, and it's not a
question you want answered differently by two copies of the same
try/except scattered across two scripts. It tries bitsandbytes' 8-bit Adam
first — each moment buffer (first and second) at roughly 1 byte/param vs
fp32 AdamW's 4 bytes/param/moment (a ~4x reduction in total optimizer state:
2 moments × 1 byte = 2 bytes/param for 8-bit vs 2 × 4 = 8 bytes/param for
fp32), which is the difference between
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
inside the training script's `main()`. The idea is straightforward once it's
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
AMD ROCm training server. It now also polls **GPU VRAM** via
`rocm-smi --showmeminfo vram` (the failure mode that actually matters for
GPU training — system-RAM polling alone can't see a VRAM OOM coming). It
parses rocm-smi's JSON output first (structured, robust) with a text-output
fallback, applies the same warn/emergency-threshold pattern as the
system-RAM check, and degrades gracefully: if `rocm-smi` isn't on PATH or
parsing fails, it logs once and skips VRAM checks (keeps the system-RAM
check working on non-ROCm boxes) rather than crashing the guard. One
design note carried over unchanged from the original: if the process being
watched has no SIGTERM handler, this is a hard, immediate kill, not a
clean save, and that's intentional — the goal is to stop before memory
pressure causes real damage, not to guarantee graceful shutdown after the
fact. Pair it with `train_cpt.py`, though, and you get the graceful case
for free: `train_cpt.py` installs its own SIGTERM handler that checkpoints
before exiting, so the two together behave as a real clean-save-then-exit
rather than a hard kill.

```
nohup bash oom_guard.sh <training_pid> [warn_mb] [emergency_mb] [poll_sec] [vram_warn_mb] [vram_emergency_mb] > oom_guard.log 2>&1 &
```

## `rocm_env.py` — AMD GPU arch detection + override

The single biggest blocker to running on "every AMD device": ROCm PyTorch
wheels are compiled for a handful of gfx architectures, and a card whose
arch isn't in that list (common on consumer RDNA1/2 cards, older
Fiji/Polaris) will import torch fine but fail at the first kernel launch
with "no kernel image is available for execution on the device." The fix is
to set `HSA_OVERRIDE_GFX_VERSION` to a compatible arch **before** the
PyTorch runtime initializes.

`rocm_env.py` does this automatically. `train_cpt.py` calls
`setup_rocm_env()` at startup (before `import torch`); it probes the GPU's
gfx arch via `rocm-smi` or `/sys/class/kfd`, compares against torch's
compiled-in arch list, and overrides only if the detected arch isn't
already supported — picking the closest same-family (`gfxNN`) arch that IS
in the list. If no family match exists, it warns loudly and doesn't
override (a wrong cross-family override can cause silent numerical errors).
You can force a specific value with `--gfx-override gfx1100`.

```python
from rocm_env import setup_rocm_env
setup_rocm_env()          # auto-detect + override if needed
import torch              # safe to import now
```

Standalone (CLI + self-test, no GPU required):
```
python3 rocm_env.py --selftest
python3 rocm_env.py --gfx-override gfx1100
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

## Troubleshooting

- **"no kernel image is available for execution on the device"** — your AMD
  GPU's gfx arch isn't in the torch wheel's compiled list. `train_cpt.py`
  calls `rocm_env.py` automatically; check its log line for what it detected
  and whether it set an override. If auto-detection didn't find a match,
  force one with `--gfx-override gfx1100` (substitute your closest family
  arch). See [`rocm_env.py`](#rocm_envpy--amd-gpu-arch-detection--override).
- **OOM dozens of steps in (not at step 0)** — almost always a silently
  missing `bitsandbytes`. The fallback to plain AdamW uses ~4x more optimizer
  memory; it allocates lazily across params, so the OOM hits later, not at
  load. Check `train_cpt.py`'s `optimizer:` log line — if it says `AdamW`
  instead of `Adam8bit`, install bitsandbytes. The Dockerfile bakes it in.
- **Optimizer mismatch on resume** — `train_cpt.py` logs whether it loaded or
  skipped the optimizer state. "skipped" means the saved and current optimizer
  classes differ (e.g. trained with bitsandbytes, resuming without it).
  Momentum restarts fresh; the step count is preserved. This is intentional
  (see `optimizer_compat_guard.py`), not a bug.
- **Vocab size silently reverts to 262144 on load** — some Gemma-4 configs
  store `vocab_size` in two places; `prune_vocab.py` fixes both by default,
  but only if you pass `--vocab-size-paths` matching your config layout.
- **`catch_and_resume.sh` hardcodes paths** — it doesn't anymore. Copy
  `config.env.example` to `config.env` and edit the values there; the script
  sources it automatically.
- **Async checkpoint write silently lost** — it can't happen silently anymore.
  `AsyncCheckpointer` now captures background-thread exceptions and re-raises
  them on the next `save()` / `wait_for_pending()`. If a write fails (disk
  full, NFS error), training stops with a real error instead of continuing
  checkpoint-less. The prior checkpoint is retained as `.prev` for recovery.

## Where this hits a real ceiling

Single-GPU throughput was the original limit here, and it isn't a rounding
error you optimize away — closing an orders-of-magnitude gap to a large
multi-trillion-token CPT target isn't a "just wait longer" problem. The
`--ddp` flag (multi-GPU via `torchrun`) gives the code path to scale to
multiple GPUs, and `--dtype fp8` / `--flash-attn` / `--compile` are the
standard per-GPU throughput levers for MI300X-class hardware — but none of
these have been benchmarked against real ROCm hardware by this repo (see the
caveat in the AMD-specific optimizations section above), so treat "lifts this
to multi-GPU" as "the plumbing exists," not "the speedup is confirmed." Even
assuming they deliver what similar techniques typically do elsewhere, the
honest framing is: targeted or bounded token budgets, domain-adapting a
pruned or expanded model, validating a pipeline end-to-end before scaling to
a full cluster. What it still isn't: a substitute for a real distributed
training framework once the token budget gets into the trillions.

## Testing

Each module with logic that can be tested without a real checkpoint ships a
`--selftest` (CPU-only, no GPU needed). Transformation tools (`prune_vocab.py`,
`prune_embeddings_torch.py`, `expand_model.py`) are covered by the pytest suite
in [`tests/`](tests/) instead — they need a real checkpoint to run their
`main()` for real, so their pure-logic functions (depth planning, orthogonal
padding shapes, merge-format parsing, the MQA layout detection) get exercised
directly instead. CI (`.github/workflows/selftest.yml`) runs both on every
push/PR.

```bash
# Run all self-tests + pytest locally (CPU-only):
for f in train_cpt.py async_checkpoint.py bnb_optimizer.py \
         local_cache_stream.py optimizer_compat_guard.py \
         rocm_env.py mtp_head.py lora_train.py train_sft.py; do
  python3 "$f" --selftest
done
pytest tests/ -v
```

`lora_train.py --selftest` needs `peft` installed (`pip install peft`, or
just `pip install -r requirements.txt`); it builds a tiny real
`transformers` model and runs real `peft` calls against it rather than
mocking either library out. `train_sft.py --selftest` checks its own
argv-rewrite logic and then delegates straight to `train_cpt.py`'s real
`self_test()` — no separate SFT-logic test exists because there's no
separate SFT-logic implementation to test.

`.gitignore` keeps the usual local-only noise (`.venv/`, `__pycache__/`,
`.pytest_cache/`, `.DS_Store`) and `config.env` specifically — the file
`catch_and_resume.sh` reads its real paths from, generated from
`config.env.example`, never meant to be committed since it'll have your
actual filesystem layout in it.

## Requirements

Confirmed in active use (see [`requirements.txt`](requirements.txt) for
tested-known-good versions, or use the [`Dockerfile`](Dockerfile) which
bakes in a ROCm torch + all deps):

- `torch` (ROCm build for AMD GPUs — install from AMD's index, not PyPI; it's
  deliberately not in `requirements.txt`)
- `safetensors`
- `numpy`
- `transformers` (confirmed working against `5.7.0`; if you're on a
  different version, check whether your `Gemma4Config` registers
  `model_type` as `"gemma4"` or `"gemma4_unified"` — `prune_vocab.py`
  handles that specific mismatch)
- `bitsandbytes` (8-bit Adam; falls back to plain AdamW at ~4x optimizer
  memory if unavailable)
- `tensorboard` (optional; for `--tb` logging — not bundled with torch, install
  separately; if absent, `--tb` warns and falls back to stdout)
- `flash-attn` (optional; for `--flash-attn` — build from source on ROCm with
  `pip install flash-attn --no-build-isolation`; falls back to standard attention)
- `torchao` (optional; for `--dtype fp8` — fp8 training on MI300X/MI325X; falls
  back to bf16 if absent)
- `peft` (only needed for `lora_train.py` — confirmed working against `0.19.1`,
  the current real PyPI release at the time this was written, checked via
  `pip index versions peft`, not guessed. `train_cpt.py` / `train_sft.py`
  don't import it at all; you don't need it unless you're specifically
  reaching for LoRA)

`requirements.txt` lists tested-known-good versions for convenience, not as a
strict constraint — if your ROCm stack needs a different torch, override it.
The only hard pin is `transformers` (the Gemma4Config model_type registration
differs across versions).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the actual ground rules, but the
short version is the same discipline this README tries to hold itself to: no
claims the code doesn't back up, no mocked self-tests, and model-family
constants go in CLI flags, not new hardcoded branches. If you add a tool,
give it a `--selftest` (or, if it's a transformation tool that needs a real
checkpoint to run, pytest coverage of its logic instead — see `tests/`
above for what that looks like in practice).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
