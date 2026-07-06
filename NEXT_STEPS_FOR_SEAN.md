NEXT STEPS — read before pushing anywhere or posting anywhere
================================================================

This directory is a PREPARE-ONLY draft. Nothing has been pushed to GitHub and
nothing has been posted to Discord or any forum. Both of those are manual
steps you do yourself, whenever you're ready — see the bottom of this file.

WHAT WAS SCRUBBED / GENERICIZED, AND WHY
-----------------------------------------

- Project name "Zacoda" removed everywhere. The showcased pipeline is
  de-branded and referred to generically as a "Gemma-4-family CPT pipeline" —
  the point of the public repo is the ROCm engineering, not the commercial
  product name.
- All ModelScope-specific details removed: no box hostnames, no account
  details, no mention of the ModelScope platform's session-reset behavior,
  no `/mnt/workspace` paths. The README just says "a single AMD MI300X GPU,"
  which is the true and sufficient claim for the pitch.
- All Tailscale / Cloudflare Tunnel access details removed — none of that
  belongs in a public repo regardless of branding.
- No HF_TOKEN, API keys, or secrets.json contents included anywhere (there
  weren't any embedded in the two source scripts to begin with, but I
  checked explicitly).
- No real name, email, or account handles included. I did not use "Sean"
  anywhere in the README or scripts; the training-lesson anecdotes are
  written in third person / passive voice ("a real run against a
  14.7B-parameter model...") instead of naming who ran it.
- Absolute local Mac paths (anything starting with `/Users/SZhang/...`)
  removed from comments/docstrings and replaced with generic relative paths
  like `./checkpoints/base_12b_pruned`.
- GCS bucket names, custom internal class names (`ZacodaPlusForCausalLM`,
  `modeling_zacoda_plus.py`), and the full `windowed_sft_cuda.py` training
  script itself were NOT copied into this repo. Only `prune_vocab.py` and
  `prune_embeddings_torch.py` (the two scripts explicitly requested) are
  included as full working code. The training script's design (optimizer
  choice, async checkpointing, local-cache streaming) is described in the
  README in my own words, without copying source that references the
  project-internal module names.

THINGS I WASN'T FULLY SURE WERE SAFE — PLEASE DOUBLE-CHECK
-------------------------------------------------------------

- I did NOT include a claim about FlashAttention-2 being used in this
  pipeline. I checked both source scripts and grepped the project's own
  pipeline_state.md for `attn_implementation` / `flash_attention_2` / `sdpa`
  and found no evidence FlashAttention-2 is actually wired into
  `windowed_sft_cuda.py` — the only FlashAttention mention in the docs is
  about a *different*, CUDA-only context unrelated to this box. Rather than
  invent a claim, I left it out entirely. If you know FlashAttention-2 (or
  ROCm's flash-attention fork) actually is wired in somewhere I didn't see,
  add it back in with the real detail — don't restore my draft's wording
  without checking first.
- The base model family is a third-party "abliterated" Gemma-4 checkpoint
  (from `huihui-ai` on Hugging Face — a public, pre-existing community
  fine-tune, not something built in this project). I didn't reference the
  specific HF repo ID or "abliterated" framing in the public README since
  it's tangential to the ROCm engineering pitch and could invite an
  off-topic discussion; if you want the community to know the exact base
  model, you can add that back deliberately.
- All numbers in the README (batch/seqlen OOM comparison, the ~110-iteration
  OOM detail, the 14.7B-param figure, the transformers 5.7.0 version, the
  bitsandbytes reinstall story) came from direct greps of
  docs/pipeline_state.md. I did not invent or round any throughput/VRAM
  figures beyond what was explicitly written there — please skim the README
  once yourself to confirm nothing reads as more precise/impressive than the
  source material actually supports.
- No LICENSE file was added — pick one before treating this as reusable.

DISCORD PITCH DRAFT (paste into AMD ROCm Developer Hub "Show and Tell" / "Projects")
---------------------------------------------------------------------------------------

Hey all — sharing a small ROCm project I've been building: a vocabulary-pruning
+ continued-pretraining pipeline for Gemma-4-family models, running end-to-end
on a single AMD MI300X.

The idea: take a 12B-parameter Gemma-4 base checkpoint, prune out vocabulary
you don't need (script-based heuristic — drops CJK/Cyrillic/Arabic/Devanagari/
Mongolian tokens plus a set of Romance/Germanic diacritic tokens, while
protecting every special/added token), slice the embedding matrix down to
match, and then run continued pretraining on the result — all on one GPU
under ROCm/PyTorch.

A few things I had to actually solve along the way that might be useful to
other people on MI300X:
- bitsandbytes 8-bit Adam vs plain AdamW: losing bitsandbytes on a fresh
  container silently falls back to fp32 AdamW, which needs ~4x the optimizer
  memory — that was enough to OOM a 14.7B-parameter run about 110 iterations
  in, once the optimizer's state had fully allocated. Now there's an explicit
  reinstall step and a resume-time guard that refuses to load one optimizer
  type's state into a different one (this used to silently corrupt memory
  usage instead of erroring cleanly).
- Async checkpointing: GPU→CPU snapshot synchronously, then the actual disk
  write happens on a background thread so training doesn't stall for the
  full write — with an atomic temp-dir-then-rename so a killed process can
  never leave a half-written checkpoint at the real path.
- Local JSONL data caching instead of live HF streaming, so a flaky network
  path to the data source doesn't stall the GPU.
- A concrete, measured batch/seqlen tradeoff: batch=2 at seqlen=1024 uses
  less memory than batch=1 at seqlen=2048 for the same tokens/step, because
  attention scales ~O(seq_len²) — moving to the smaller-seqlen/larger-batch
  config turned two repeated OOMs into a stable run.

Repo: [add your GitHub URL here once you've pushed it]

Where I'm hitting a wall: single-GPU throughput is a real ceiling, not
something more optimization fixes — the pipeline works well for bounded CPT
budgets, but scaling the token count meaningfully needs more than one GPU.
That's the actual ask behind sharing this: looking to scale this same
pipeline across multiple MI300X GPUs, and would value any ROCm-side guidance
on multi-GPU training setups (FSDP/DeepSpeed on ROCm, NCCL/RCCL topology
tips, anything people here have found that actually works) — and if there's
a path to compute credits for continuing this as a public ROCm reference
pipeline, I'd love to talk.

Happy to answer questions about any of the above.

MANUAL STEPS YOU STILL NEED TO DO
------------------------------------

1. Review every file in this directory yourself — especially the two
   "wasn't fully sure" items above.
2. Decide on a license and add a LICENSE file if you want this to be
   genuinely reusable.
3. Create the actual GitHub repo yourself (this task did not touch
   github.com in any way) and push this local repo to it yourself.
4. Post to Discord / any forum yourself, using the draft above as a
   starting point — no message was sent anywhere by this task.
