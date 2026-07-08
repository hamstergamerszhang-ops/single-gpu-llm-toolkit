# single-gpu-llm-toolkit

Nineteen independently-runnable tools (seventeen Python + two shell) for
adapting an LLM checkpoint on a **single AMD GPU** under ROCm/PyTorch — no
multi-node cluster, no distributed training framework. The pipeline covers
the full path from a base checkpoint to a trained one: shrink a tokenizer,
grow a model's width and depth, continue-pretrain or fine-tune it, and
survive the specific ways a single GPU fails you along the way — OOM,
crashes, a data source that goes unreachable mid-run. `preprocess_data.py`,
`benchmark.py`, and `generate.py` round out the pipeline end-to-end: clean
data in, compare configs before committing to a long run, and actually talk
to the model you just trained. Every tool ships its own `--selftest` or
pytest coverage (the two shell scripts, `catch_and_resume.sh` and `oom_guard.sh`, are not covered by selftests or pytest — their logic is exercised manually; see [Testing](#testing)) and is independently runnable —
use the whole pipeline or just the one tool that solves your problem.

All training here is full-parameter fine-tuning (`train_cpt.py` /
`train_sft.py`) — no LoRA, no adapters. That's a deliberate choice, not an
oversight: the quality this repo's own runs are built around comes from
training the real weights, not a low-rank approximation of them.

None of this is pinned to one GPU. There's no device-name check, no
architecture branch, no VRAM-threshold logic in the source —
batch size, sequence length, and how many layers stay unfrozen are all
plain CLI flags, so the same scripts scale down to a smaller card by
freezing more layers and shrinking the batch, or scale up by unfreezing
more and running bigger batches. Standard ROCm/PyTorch throughout — nothing
here calls out to an MI300X-only code path. It happens to have been built
and run on one MI300X; there's nothing in it that ties it there.

Here's what that actually buys you, concretely, instead of just abstractly:
you don't have to choose between "fits in memory" and "actually trains all
the weights." `--start`/`--end` freeze everything outside a layer window
(`apply_window_freeze()` in `train_cpt.py`), so a card that can't hold
gradients + optimizer state for the whole model can still do real,
full-parameter training on a slice of it — no LoRA, no adapters, no
low-rank approximation standing in for the actual weights, just less of
the model unfrozen at once. Combine that with gradient checkpointing (on
by default) and the 8-bit-Adam-with-AdamW-fallback optimizer
(`bnb_optimizer.py`) and the same lever works in both directions: freeze
more and shrink the batch on a smaller card, or unfreeze everything and
run it wide on an 80GB+ one.

The other half of "actually usable on hardware that isn't a managed
cluster" is what happens when the run dies. `train_cpt.py` checkpoints
every `--checkpoint-every` steps (500 by default) to local disk, and the
write itself is atomic — `train_cpt.py`'s `atomic_save_checkpoint()` (or the opt-in `async_checkpoint.py`) builds the new checkpoint
in a temp directory, then does two `os.replace()` calls (retire the live
checkpoint to `.prev`, promote the temp dir to live), so a `kill -9` or a
host reset mid-write can never leave you with a half-written, unreadable
checkpoint; worst case it lands you on `.prev`, the last *good* one, not
a corrupt new one. On restart, `train_cpt.py` doesn't need a `--resume`
flag or you to remember a step number — it just checks whether
`<save_dir>/training_state.pt` exists and picks up from there
automatically. Practically: if your box resets mid-run, what you lose is
whatever training happened since the last checkpoint boundary — at most
`--checkpoint-every` steps of work, not the run. That number is steps, not
minutes, on purpose — how long 500 steps takes depends on your model size,
batch, and card, so there's no single wall-clock figure that holds across
setups; for your own hardware, `benchmark.py` reports step time directly, or
just watch `train_cpt.py`'s own iter logging.

That resume mechanism is passive, though — it makes coming back from a
crash cheap, it doesn't do the coming-back for you. `catch_and_resume.sh`
is the part that does: it's a supervisor loop that actually relaunches
`train_cpt.py` after any exit (crash, OOM-kill, preemption) and keeps
doing so until the target iteration count is reached, a stop-file shows
up, or the same-position-retry cap is hit (so a genuinely broken config
doesn't retry forever, silently burning GPU-hours on a run that was never
going to succeed). It also keeps a loss-tagged checkpoint history and
rolls back to the best one on a loss regression, so a bad data patch
doesn't get compounded checkpoint after checkpoint. `oom_guard.sh` is
narrower and honest about it: it only watches memory and VRAM and sends a
clean `SIGTERM` before a hard OOM can take the driver down with it — it
does not relaunch anything itself. Run it standalone and a killed
training process just stays dead; pair it with `catch_and_resume.sh` and
the two together give you the actual unattended story: guard kills
cleanly before real damage, checkpoint is already safe on disk from the
SIGTERM handler, supervisor notices the exit and brings training back up
on its own.

Every model-specific constant — the embedding tensor's key name, the
vocab_size config path, the layer-naming prefix, the sharding size, the
depth/width step sizes, the GQA head count — is a CLI flag, defaulting to
the Gemma-4 layout these tools were built against but pointable at whatever
your own checkpoint actually uses. The one piece that isn't a flag is
`expand_model.py`'s submodule key *suffixes*
(`gate_proj`/`up_proj`/`down_proj`, `q_proj`/`k_proj`/`v_proj`/`o_proj`),
because those names are shared by most Llama-derived decoder architectures
(Llama, Mistral, Qwen2/3, every Gemma generation) — verified directly
against the installed `transformers` library's modeling source. They're not
universal: GPT-2, the original Phi, Phi-3, Falcon, MPT, and BLOOM use
different, fused-QKV naming and need code changes (not a flag) to support.
`expand_model.py`'s GQA fix runs a detection pass against the loaded
checkpoint's real tensors before touching anything, and skips cleanly with
a logged reason if the checkpoint doesn't match the MQA layout it targets,
instead of assuming every input does.

You don't need to use these together or in order — each one solves a
different single-GPU problem on its own. The canonical pipeline order, if
you use them together, is:

```
[preprocess_data.py] → prune_vocab.py → prune_embeddings_torch.py → expand_model.py → [mtp_head.py] → train_cpt.py / train_sft.py → [generate.py]
```

Skip whichever steps you don't need — data preprocessing, pruning, expansion,
and MTP are all independently optional, and training directly against an
unmodified base checkpoint with raw JSONL is a completely normal way to use
`train_cpt.py` (or `train_sft.py`, its SFT-only alias — see below) on its
own. `benchmark.py` sits outside this chain entirely — run it against any
checkpoint, at any point, to measure a config rather than train or generate
from it.

## Table of Contents

- [Installation](#installation)
- [Tools](#prune_vocabpy--shrink-a-tokenizer-you-dont-need-in-full)
  - [`prune_vocab.py`](#prune_vocabpy--shrink-a-tokenizer-you-dont-need-in-full)
  - [`prune_embeddings_torch.py`](#prune_embeddings_torchpy--apply-that-vocab-cut-to-the-actual-weights)
  - [`expand_model.py`](#expand_modelpy--grow-a-models-width-and-depth-without-retraining-from-scratch)
  - [`mtp_head.py`](#mtp_headpy--add-a-multi-token-prediction-head)
  - [`train_cpt.py`](#train_cptpy--continued-pretraining-single-gpu)
  - [`train_sft.py`](#train_sftpy--the-sft-only-name-for-train_cptpy)
  - [`catch_and_resume.sh`](#catch_and_resumesh--keep-a-single-gpu-run-alive-across-crashes)
  - [`preprocess_data.py`](#preprocess_datapy--dedup-filter-and-pack-training-data)
  - [`benchmark.py`](#benchmarkpy--measure-throughput-and-vram-across-configs)
  - [`generate.py`](#generatepy--streaming-inference-from-a-trained-checkpoint)
  - [`compress_model.py`](#compress_modelpy--quantize-any-model-int8int4fp8)
  - [`tensor_parallel.py`](#tensor_parallelpy--auto-detect-multi-gpu-run-with-pipeline-parallelism)
  - [`smart_hipify.py`](#smart_hipifypy--intelligent-cudahip-converter)
  - [Standalone utilities](#standalone-utilities)
  - [`rocm_env.py`](#rocm_envpy--amd-gpu-arch-detection--override)
- [Tips](#tips)
- [Troubleshooting](#troubleshooting)
- [Where this hits a real ceiling](#where-this-hits-a-real-ceiling)
- [Testing](#testing)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

**Option 1: Docker (recommended — bakes in all deps including ROCm torch):**

```bash
docker build -t single-gpu-llm-toolkit .
docker run --device /dev/kfd --device /dev/dri --group-add video \
           --shm-size 64G -v $(pwd):/work -w /work -it single-gpu-llm-toolkit \
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
they're flag-driven rather than hardcoded. Some Gemma-4 configs store
`vocab_size` in two places — a top-level field and a nested one — and
missing either one silently reverts the vocab size at load time; this
script fixes both by default (`--vocab-size-paths` if your own config nests
it somewhere else). A second, narrower fix renames
`model_type: "gemma4_unified"` to `"gemma4"` when a checkpoint's
`config.json` uses the older string but the installed `transformers`
registers `Gemma4Config` under the shorter name — it only fires on that
exact match, so it's a no-op against any other model family's config.
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
it back — is a plain key lookup and tensor slice with no architecture-specific
logic. Point `--embed-key` at whatever your own checkpoint's safetensors
header actually calls its embedding weight (e.g. plain
`model.embed_tokens.weight` on many non-Gemma architectures) and it works
the same way. Useful standalone any time you've already got a vocab remap
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
otherwise just trading away model quality for a saving you don't need.
Before rewriting anything, it checks the loaded checkpoint's real tensors —
does a `v_proj` key actually exist for these layers (it shouldn't, if the
fix applies), does `k_proj`'s real shape agree with what the config's
kv-head count claims, does the config even carry a resolvable `head_dim` at
all — and skips the pass cleanly with a specific reason logged if any check
fails, instead of running anyway and either crashing on a shape mismatch or
silently overwriting a real, already-trained V projection on an
architecture that has one. `--force-gqa-fix` overrides a failed layout
check for the case where you've verified by hand that the fix is still
correct; it does not override "there's no head_dim to compute a shape
from at all," which fails with a clear error instead of crashing deeper in
the tensor arithmetic.

Uses PyTorch + numpy + safetensors (no Apple-Silicon-only MLX dependency).
The width/depth expansion logic and the tensor-key *prefix* are both
parameterized. The tensor-key *suffixes*
(`gate_proj`/`up_proj`/`down_proj`, `q_proj`/`k_proj`/`v_proj`/`o_proj`) are
not parameterized because they're shared by most Llama-derived decoder
architectures already (see the intro above); on an architecture with
different suffixes (fused-QKV models like GPT-2, Phi-3, Falcon, MPT, BLOOM),
retargeting needs the submodule key edits noted in the module docstring, not
just a flag.

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

**Modeling code:** `modeling_custom.py` (repo root) is a stub `CustomForCausalLM`
that loads these weights with zero missing/unexpected keys and runs a
structurally-correct forward pass — it does NOT implement the real MTP
training loss (target shifting, weighted loss sum); see that file's own
docstring for exactly what it does and doesn't do. Copy it alongside the
checkpoint mtp_head.py writes (config.json's `auto_map` already points at it)
and extend `forward` for your real train/inference path. It consumes the keys
`mtp_head.py` documents: `model.mtp_layers.{i}.enorm.weight`, `.eh_proj.weight`,
`.block.<suffix>`, `.lnorm.weight`, `model.mtp_layers.norm.weight` (the shared
final norm — note the key lives under `mtp_layers`, not a separate `mtp`
prefix).

```
python3 mtp_head.py --src <expanded_checkpoint> --dst <mtp_checkpoint>
```

## `train_cpt.py` — continued pretraining, single GPU

The CUDA/ROCm training loop, and the standalone entry point for training any
checkpoint — pruned, expanded, or neither. It handles layer-window
freeze/unfreeze (full-model training when VRAM allows, partial-layer
windowing when it doesn't), gradient checkpointing, an 8-bit-Adam-with-AdamW-
fallback optimizer, async local-disk checkpointing (opt-in via
`--async-checkpoint`, off by default), a local-JSONL data/cache mode, and a
clean SIGTERM-triggered checkpoint-and-exit.

```
python3 train_cpt.py --model <checkpoint> --data <jsonl_dir_or_file> --save <out_dir> --batch 1
```

Optimizer construction, async checkpoint writes, the optimizer-type resume
guard, and local-cache data streaming each live in their own standalone
module (see [Standalone utilities](#standalone-utilities) below);
`train_cpt.py` imports and calls them rather than duplicating the logic.

### AMD-specific optimizations

`train_cpt.py` supports several AMD-ROCm-specific optimizations, each behind
a CLI flag. `--flash-attn`, `--dtype fp8`, and `--compile` fall back to the
default path (bf16, eager, standard attention) if their dependency isn't
installed; `--ddp` and `--profile` are infrastructure flags that either run
or don't, based on whether you pass them. The relative speedups quoted below
for flash-attn/fp8/compile are the figures generally reported for these
techniques on modern accelerators — measuring the actual multiplier on your
own card and workload is what [`benchmark.py`](#benchmarkpy--measure-throughput-and-vram-across-configs)
is for:

- **`--flash-attn`** — Flash Attention 2. Reduces attention VRAM from
  `O(seqlen²)` to `O(seqlen)`, which is the mechanism that speeds up
  long-context training. Requires `flash-attn` built for ROCm (`pip install
  flash-attn --no-build-isolation`). Falls back to standard attention with a
  warning if not installed.
- **`--dtype fp8`** — fp8 training via `torchao`'s `Float8Linear`
  (`float8_e4m3fn`). MI300X, MI300A, AND MI325X all have native fp8 compute —
  verified directly against AMD's own gpu-arch-specs documentation, all three
  are `gfx942` (there is no separate `gfx940`/`gfx941` architecture for
  MI300A/MI325X; those strings appear in some early/internal ROCm references
  but don't correspond to real, currently-shipping distinct chips — don't
  reintroduce a "gfx940=MI300A, gfx941=MI325X" mapping in a future pass). A
  runtime capability gate checks `torch.cuda.get_device_capability()` and
  gracefully falls back to bf16 with a warning on any card outside gfx942
  (e.g. gfx1100 / RX 7900 XTX), so `--dtype fp8` is safe to pass on any AMD
  card — it will just skip fp8 on unsupported hardware. Falls back to bf16 if
  `torchao` is missing.
- **`--compile`** — `torch.compile()` with ROCm's inductor backend for kernel
  fusion + graph optimization. First few steps are slower (compilation), then
  faster. Falls back to eager mode if compilation fails. With `--pack` (fixed
  sequence lengths), `dynamic=False` is used to avoid recompilation thrash;
  without `--pack`, dynamic shapes are enabled so variable-length inputs work.
- **`--compile-mode <mode>`** — selects the `torch.compile` mode: `default`,
  `reduce-overhead`, or `max-autotune` (default). `max-autotune` spends more
  time upfront autotuning but yields the best steady-state ROCm throughput;
  `reduce-overhead` is better for small models or short runs. Ignored unless
  `--compile` is set.
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
  train_cpt.py --ddp ...`. Only rank 0 writes checkpoints/logs and runs
  held-out eval; all ranks participate in gradient all-reduce. Rank 0's
  checkpoint and eval calls unwrap the DDP wrapper before running (see
  `unwrap_ddp()`), so those rank-0-only forward passes never trigger a
  collective that other ranks would need to join. Developed and its logic
  verified on a single GPU plus a CPU-only multi-process
  `torch.distributed` harness (no real multi-GPU ROCm cluster in the loop);
  run your own convergence check the first time you point it at a real
  multi-GPU box.
- **`--fsdp`** — multi-GPU training via `FullyShardedDataParallel` (FSDP).
  Unlike `--ddp` (which replicates the full model on every GPU), FSDP
  **shards** params + grads + optimizer state across GPUs — so a model too
  large for one MI300X (e.g. a 27B+ model) can be trained across a node of
  them. FSDP also avoids the `find_unused_parameters=True` hazard that `--ddp`
  hits with windowed `--start`/`--end` freezing (FSDP handles frozen params
  natively). Launch with `torchrun --nproc_per_node=N train_cpt.py --fsdp
  ...`. Checkpoint saving gathers the full (unsharded) state dict to rank 0
  via `FSDP.state_dict_type(..., FULL_STATE_DICT)` before writing, so saved
  checkpoints are standard HF format and loadable on a single GPU. Use
  `--sharding-strategy` to control the shard granularity:
  - `full` (default, `FULL_SHARD`): shards params+grads+optimizer state.
    Maximum memory savings, most communication.
  - `shard-grad-op` (`SHARD_GRAD_OP`): shards grads+optimizer state only;
    params stay replicated. Less comm overhead, more memory.
  - `no-shard` (`NO_SHARD`): equivalent to DDP.
  
  Multi-node launch (FSDP or DDP):
  ```
  # Single-node, 8 GPUs:
  torchrun --nproc_per_node=8 train_cpt.py --fsdp --model ... --save ...

  # Multi-node (2 nodes x 8 GPUs each), using one node as rendezvous host.
  # --rdzv-id must be the SAME on all nodes so they find each other:
  # Node 0 (rendezvous host):
  torchrun --nnodes=2 --nproc_per_node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=node0:29500 --rdzv-id=cpt-multinode \
      train_cpt.py --fsdp --model ... --save ...
  # Node 1 (identical command, same --rdzv-id and --rdzv-endpoint):
  torchrun --nnodes=2 --nproc_per_node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=node0:29500 --rdzv-id=cpt-multinode \
      train_cpt.py --fsdp --model ... --save ...
  ```
  FSDP requires `use_reentrant=False` gradient checkpointing (set
  automatically when `--fsdp` is used); the reentrant variant is
  incompatible with FSDP's forward hooks. `--fsdp` and `--ddp` are mutually
  exclusive.
- **`--accum N`** (alias `--gradient-accumulation-steps`) — accumulate
  gradients over `N` micro-batches of size `--batch` before each optimizer
  step, giving an effective batch size of `batch * accum` without the extra
  VRAM a literally-larger `--batch` would need. Each micro-batch's loss is
  divided by `N` before `backward()`, so the accumulated gradient matches
  what a real batch of size `batch * accum` would produce. Default `1` (no
  accumulation). Under `--ddp` or `--fsdp`, the non-final micro-batches run
  inside `model.no_sync()`, so only the last micro-batch's backward triggers
  the gradient sync (all-reduce for DDP, reduce-scatter+all-gather for FSDP)
  — `N` micro-batches cost one sync, not `N`.

These flags compose (`--fsdp --flash-attn --dtype fp8 --compile` on a
multi-GPU MI300X box); `benchmark.py` is the way to measure what that
combination actually gets you on your own hardware before committing a long
run to it.

## `train_sft.py` — the SFT-only name for `train_cpt.py`

`train_cpt.py` already does SFT. It's the default mode — omit `--cpt` and you get chat-template tokenization
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

## `preprocess_data.py` — dedup, filter, and pack training data

The rest of this pipeline assumes your JSONL is already reasonably clean, and
that assumption doesn't hold for most real data sources. This is the tool
that gets it there before it ever reaches `train_cpt.py`: exact dedup (drops
rows whose text field is identical to one already seen — no fuzzy/minhash
dedup, on purpose. Approximate dedup quality varies wildly by dataset, and a
false-positive drop silently throws away real training data with no way to
notice; if you need fuzzy dedup, run a dedicated tool like `datasketch`
upstream of this one), length filtering (`--min-chars`/`--max-chars` for a
fast tokenizer-free pass, or `--min-tokens`/`--max-tokens` if you'd rather
pay for exactness with a real `--tokenizer`), script filtering
(`--drop-scripts cjk,arabic,...`, reusing `prune_vocab.py`'s `classify()` —
the same character heuristic, not real language ID, so it catches distinctive
non-Latin script characters and nothing more subtle than that), and sequence
packing (combines short rows into sequences up to `--pack-seqlen` chars,
cutting the padding waste that `train_cpt.py`'s own `--pack` flag then
compresses further at collation time). `--dry-run` prints the stats — rows
in, dropped by reason, sequences packed — without writing anything, which is
worth doing once on new data before trusting the real run.

```
python3 preprocess_data.py --src data.jsonl --dst filtered.jsonl \
    --min-chars 50 --max-chars 10000 --drop-scripts cjk,arabic --pack-seqlen 2048
```

## `benchmark.py` — measure throughput and VRAM across configs

Guessing at batch size and sequence length on a new card wastes real
GPU-hours on trial and error. This tool replaces the guessing: it loads the
model, runs real forward + backward + optimizer-step iterations against
random input (dummy data, so no dataset needed — this measures the model
and the hardware, not what you feed it), and reports tokens/sec, peak VRAM,
and average step time. Pass several configs in one invocation
(`batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8`) and it prints
a comparison table, so "does fp8 actually help on this card" or "is
batch=4/seqlen=512 faster than batch=2/seqlen=1024 for the same token count"
become a five-minute measurement instead of a guess baked into a multi-day
run.

```
python3 benchmark.py --model ./checkpoints/base_expanded_15b \
    --configs "batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8"
```

### Generation benchmark (`--gen`)

For inference-focused metrics, `--gen` benchmarks generation throughput
instead of training. It measures **TTFT** (time-to-first-token, prefill
latency), **TPOT** (time-per-output-token, decode latency), total time, and
tokens/sec for a single prompt:

```
python3 benchmark.py --model ./checkpoints/base_expanded_15b --gen \
    --gen-prompt-len 2048 --gen-len 256
```

## `generate.py` — streaming inference from a trained checkpoint

Training produces a checkpoint; this is what actually talks to it. It loads
a checkpoint the same way `train_cpt.py` does, then generates with a real
`TextIteratorStreamer` so tokens print as they're produced instead of all at
once at the end of a potentially long generation — worth calling out because
`TextIteratorStreamer` doesn't stream on its own; it just pushes decoded text
onto a queue as `generate()` produces it, and something still has to run
`generate()` on a background thread while the main thread drains that queue,
or nothing prints until generation is already finished and the streamer's
already seen the whole response go by unread. This runs the same
AMD-specific flags as `train_cpt.py` (`--flash-attn`, `--dtype fp8`,
`--compile`, `--gfx-override`, `--hip-alloc-conf`), so the "runs on any
ROCm-capable AMD GPU" claim carries over to inference, not just training.
KV-cache is on by default here — the opposite of training's
`use_cache=False` — because generation without a cache is the one case where
that would actually cost you real wall-clock time, recomputing every prior
token's attention on every new token. Interactive mode (type prompts,
Ctrl+D to exit) or batch mode (`--input prompts.txt`, one prompt per line).

```
python3 generate.py --model ./checkpoints/model_cpt_1 --flash-attn --dtype fp8
```

## `compress_model.py` — quantize any model (int8/int4/fp8)

Takes any HuggingFace-format checkpoint and produces a quantized version using
torchao. Supports int8 (~2x smaller, negligible loss), int4 (~4x smaller,
minimal loss), and fp8 (~2x smaller, best on MI300X). All three work on ANY
AMD card — the weights are dequantized to bf16 for the matmul, so no special
hardware is required (fp8 gets native-speed matmuls on MI300X/MI325X, but
still works on everything else). Auto-detects nested (Gemma-4) vs flat
(Llama/Mistral/Qwen) config layouts.

```
python3 compress_model.py --src ./checkpoints/base_15b --dst ./checkpoints/base_15b_int4
```

## `tensor_parallel.py` — auto-detect multi-GPU, run with pipeline parallelism

Detects how many AMD GPUs are available and distributes the model across them
using an explicit, layer-balanced device map (pipeline parallelism — different
layers on different GPUs, with activations passed between them). On a node of
identical GPUs (e.g. 8x MI300X), this balanced split is better than HF's
`device_map="auto"` greedy memory fit, which can leave one GPU underloaded and
another holding the LM head + embeddings + last layers (a large pipeline
bubble). Use `--device-map auto` to fall back to HF's automatic assignment, or
`--device-map single` to force single-GPU. If only 1 GPU is detected, falls
back to single-GPU generation automatically. Supports interactive prompts or
batch mode (`--input prompts.txt`), flash attention (`--flash-attn`), and
streaming token-by-token output. Uses `rocm_env.setup_rocm_env()` for the gfx
override, so the "every AMD device" gfx-override auto-detection carries over.

```
python3 tensor_parallel.py --model ./checkpoints/base_15b
```

## `smart_hipify.py` — intelligent CUDA→HIP converter

Smarter than AMD's stock `hipify-perl`. Does the same API name substitutions
(`cudaMalloc`→`hipMalloc`, etc.) but ALSO: detects CUDA library calls
(cuBLAS/cuDNN/cuSPARSE) that have NO drop-in HIP equivalent and flags them with
`/* HIPIFY: TODO */` comments instead of silently producing broken code; auto-adds
`#include <hip/hip_runtime.h>`; warns about CUDA headers with no HIP mapping;
counts `__global__` kernels; and produces a full diff-style report of every
change. The "smart" part is being honest about what can and can't be automated.

```
python3 smart_hipify.py --src kernel.cu --dst kernel.cpp
python3 smart_hipify.py --src ./cuda_project/ --dst ./hip_project/ --recursive
```

## Standalone utilities

Four of these back `train_cpt.py` directly and are independently useful on
their own merits: optimizer construction, async checkpoint writes, the
optimizer-type resume guard, and local-cache data streaming. A fifth,
`oom_guard.sh`, is a memory-safety guard whose poll/warn/kill pattern is generic (the implementation uses Linux `/proc/meminfo` and ROCm's `rocm-smi`).

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
warning. The fallback's failure mode isn't a crash at step 0 — it's an OOM
dozens of iterations in, once the
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

**`async_checkpoint.py`** is the background-thread checkpoint writer used by
`train_cpt.py`. Serializing tens of GB to a possibly-slow disk or NFS mount
is slow, and there's no reason the GPU should sit idle waiting for it. The
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
same row order repeat forever. `train_cpt.py`'s `--cpt-cache` flag imports
`stream_from_cache` from here directly.

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
nohup bash oom_guard.sh <training_pid> [warn_free_mb] [emergency_free_mb] [poll_sec] [vram_warn_mb] [vram_emergency_mb] > oom_guard.log 2>&1 &
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

## Tips

- **Reinstall `bitsandbytes` explicitly on every fresh container.** It's easy
  to lose silently on a rebuild, and the failure mode isn't a crash at step 0
  — it's an OOM dozens of iterations in, once the ~4x-larger fallback AdamW
  optimizer state has fully allocated. Check `train_cpt.py`'s `optimizer:`
  log line if a run OOMs later than expected.
- **Checkpointing is local-disk only, no cloud object store.** For
  cross-instance durability, sync the checkpoint directory out on your own
  schedule (a periodic rsync, say) — there's no in-process cloud upload.
- **`train_cpt.py`'s optional local-JSONL cache mode** (`--cpt-cache`) trains
  with zero network dependency once the cache is built, cycling it
  indefinitely — useful when live streaming depends on a network path you
  don't fully trust for a multi-day run.
- **Resuming across a different optimizer type is guarded, not silently
  accepted.** Loading fp32 AdamW state into a bitsandbytes Adam8bit instance
  (or the reverse) inflates memory past what the current optimizer needs and
  OOMs on the first forward pass. `train_cpt.py` checks the saved optimizer's
  class before loading (via `optimizer_compat_guard.py`) and skips the
  optimizer state — restarting momentum, keeping the step count — on a
  mismatch.
- **Test both directions of the batch-size-vs-seqlen tradeoff.** Attention's
  `O(seqlen²)` scaling means the same total tokens/step can use very
  different amounts of memory depending on how you split batch vs. sequence
  length — a smaller batch at a longer sequence length isn't automatically
  cheaper, or automatically more expensive, than the reverse. `benchmark.py`
  turns that comparison into a five-minute measurement instead of a guess
  baked into a multi-day run.
- **`TextIteratorStreamer` needs a background thread, or it prints nothing
  until generation finishes.** It only pushes decoded text onto a queue as
  `generate()` produces it; something has to drain that queue on a separate
  thread while `generate()` runs, or the "streaming" output arrives all at
  once at the end. `generate.py` runs `generate()` on a `threading.Thread`
  and iterates the streamer on the main thread.
- **`torchao`'s fp8 weight-only quantization API differs by version:** older
  releases expose a `float8_weight_only()` function; newer ones replaced it
  with a `Float8WeightOnlyConfig` class passed to `quantize_()`.
  `requirements.txt` pins `torchao>=0.5.0` with no upper bound, so either
  could be installed — `generate.py` tries the function
  first and falls back to the config class, so `--dtype fp8` works across
  that version range rather than only whichever API happened to be current
  when the code was written.

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
  store `vocab_size` in two places; `prune_vocab.py` fixes both by default
  (the default `--vocab-size-paths` covers both Gemma-4 locations). Only pass
  `--vocab-size-paths` if your config nests `vocab_size` somewhere non-standard.
- **Configuring `catch_and_resume.sh` for your own run** — copy
  `config.env.example` to `config.env` and edit the values there; the script
  sources it automatically rather than requiring edits to the script itself.
- **Async checkpoint write failed** — `AsyncCheckpointer` captures
  background-thread exceptions and re-raises them on the next `save()` /
  `wait_for_pending()` call, so a failed write (disk full, NFS error) stops
  training with a real error instead of continuing checkpoint-less. The
  prior checkpoint is retained as `.prev` for recovery.
- **Multi-GPU hang / "NCCL communicator init" error** — ROCm's NCCL (RCCL)
  sometimes needs environment variables for multi-GPU to work on certain node
  topologies. If `--ddp` or `--fsdp` hangs at the first all-reduce:
  - `NCCL_SOCKET_IFNAME=hsn0` (or `eth0`, `ibs5` — the high-speed interface on
    your node; use `ip addr` to find it). Without this NCCL may pick a slow
    management interface and hang.
  - `NCCL_P2P_DISABLE=1` — disables peer-to-peer (XGMI) if it's flaky on
    your topology. Slower but unblocks.
  - `NCCL_IB_DISABLE=1` — disables InfiniBand if present and problematic.
  - `NCCL_DEBUG=INFO` — logs NCCL's topology discovery, useful for diagnosing.
  Set these in the environment before `torchrun`:
  `NCCL_SOCKET_IFNAME=hsn0 torchrun --nproc_per_node=8 train_cpt.py --fsdp ...`

## Where this hits a real ceiling

Single-GPU throughput is the natural ceiling here — closing an
orders-of-magnitude gap to a large multi-trillion-token CPT target isn't a
"just wait longer" problem. `--ddp` scales training to multiple GPUs via
`torchrun`, and `--dtype fp8` / `--flash-attn` / `--compile` are the
standard per-GPU throughput levers for MI300X-class hardware; `benchmark.py`
is how you measure what they're actually worth on your own hardware and
workload before committing a long run to a particular combination. This
toolkit fits targeted or bounded token budgets, domain-adapting a pruned or
expanded model, and validating a pipeline end-to-end before scaling further
— it is not a distributed training framework once the token budget gets
into the trillions.

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
         rocm_env.py mtp_head.py train_sft.py \
         preprocess_data.py benchmark.py generate.py \
         compress_model.py tensor_parallel.py smart_hipify.py; do
  python3 "$f" --selftest
done
pytest tests/ -v
```

`train_sft.py --selftest` checks its own argv-rewrite logic and then
delegates straight to `train_cpt.py`'s real `self_test()` — no separate
SFT-logic test exists because there's no separate SFT-logic implementation
to test.

`.gitignore` keeps the usual local-only noise (`.venv/`, `__pycache__/`,
`.pytest_cache/`, `.DS_Store`) and `config.env` specifically — the file
`catch_and_resume.sh` reads its real paths from, generated from
`config.env.example`, never meant to be committed since it'll have your
actual filesystem layout in it.

## Requirements

See [`requirements.txt`](requirements.txt) for pinned versions, or use the
[`Dockerfile`](Dockerfile), which bakes in a ROCm torch + all deps:

- `torch` (ROCm build for AMD GPUs — install from AMD's index, not PyPI; it's
  deliberately not in `requirements.txt`)
- `safetensors`
- `numpy`
- `transformers==5.7.0` — a hard pin, not a suggestion: Gemma-4 support
  (`Gemma4Config` / `model_type="gemma4"`) doesn't exist at all in the `4.x`
  line and doesn't land until partway through the `5.x` line; `5.7.0` is the
  version confirmed to register it correctly. If you're on a different `5.x`
  version, check whether your `Gemma4Config` registers `model_type` as
  `"gemma4"` or `"gemma4_unified"` — `prune_vocab.py` handles that specific
  mismatch.
- `bitsandbytes` (8-bit Adam; falls back to plain AdamW at ~4x optimizer
  memory if unavailable)
- `tensorboard` (optional; for `--tb` logging — not bundled with torch, install
  separately; if absent, `--tb` warns and falls back to stdout)
- `flash-attn` (optional; for `--flash-attn` — build from source on ROCm with
  `pip install flash-attn --no-build-isolation`; falls back to standard attention)
- `torchao` (optional; for `--dtype fp8` — fp8 training on MI300X/MI325X; falls
  back to bf16 if absent)

`requirements.txt` lists tested-known-good versions for convenience, not as a
strict constraint — if your ROCm stack needs a different torch, override it.
The only hard pin is `transformers` (the Gemma4Config model_type registration
differs across versions).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the ground rules. Short version:
no claims the code doesn't back up, no mocked self-tests, and model-family
constants go in CLI flags, not new hardcoded branches. If you add a tool,
give it a `--selftest` (or, if it's a transformation tool that needs a real
checkpoint to run, pytest coverage of its logic instead — see `tests/`
above for what that looks like in practice).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
