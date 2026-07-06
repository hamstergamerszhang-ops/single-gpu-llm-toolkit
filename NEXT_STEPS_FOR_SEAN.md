NEXT STEPS — read before pushing anywhere or posting anywhere
================================================================

This directory is a PREPARE-ONLY draft. Nothing has been pushed to GitHub and
nothing has been posted to Discord or any forum. Both of those are manual
steps you do yourself, whenever you're ready — see the bottom of this file.

WHAT'S IN THE REPO NOW
-----------------------

All four pipeline stages are now present as real, working code, not just
described in prose:

- `prune_vocab.py` + `prune_embeddings_torch.py` — unchanged from the prior
  draft (tokenizer pruning + embedding tensor surgery).
- `expand_model.py` — NEW. Width+depth expansion (orthogonal-QR width init,
  zero-init depth duplication, optional GQA fix), ported from
  `expand_plus_15b_torch.py`. The `add_mtp_modules()` piece of the original
  (which instantiates a project-specific Multi-Token-Prediction module class)
  was NOT carried over, since it depends on a custom modeling file that isn't
  in this repo — the `--mtp-depths` flag still sets the matching config
  fields (harmless if your architecture doesn't define an MTP module class)
  but doesn't try to build MTP weights itself. If you want that piece
  included for real, it needs your actual custom modeling class alongside it.
- `train_cpt.py` — NEW. The CUDA/ROCm training loop, ported from
  `windowed_sft_cuda.py`. Carried over: layer-window freeze/unfreeze, LR
  warmup→cosine schedule, SFT + CPT modes, atomic + async checkpointing, the
  bitsandbytes/AdamW fallback, the cross-optimizer resume guard, local-cache
  streaming (`--cpt-cache`). Deliberately NOT carried over: QAT (quantization-
  aware training), live HF streaming (`--stream`/`PrefetchBuffer`), and MTP
  loss handling — all three depend on project-specific modules
  (`cpt_streaming_data.py`, the MTP module class) that aren't in this repo.
  If you want those back, they need to be ported in with their real
  dependencies, not stubbed.
- `catch_and_resume.sh` — NEW. A crash-recovery supervisor for `train_cpt.py`,
  written against that script's REAL flags (checked directly against its
  argparse section, not assumed). Important nuance documented in the script's
  own header: this is NOT a copy of the older MLX-side supervisor
  (`run_cpt_with_autoresume.sh` in the source project) — that script passed
  explicit `--resume`/`--start-iter` flags because its target script's
  checkpoint filenames reset their step counter every invocation.
  `train_cpt.py` self-resumes from `<save>/training_state.pt` automatically,
  so this supervisor just relaunches the same command — but it adds its own
  loss-tagged checkpoint history + rollback-on-regression + stall detection
  on top, since `train_cpt.py`'s own `--save` path is a single slot that a
  bad training patch could otherwise silently overwrite.
- `README.md` — rewritten to cover all four scripts, the real GCS-vs-local-
  disk story (see below), and the local-cache-vs-streaming rationale.

WHAT WAS SCRUBBED / GENERICIZED, AND WHY
-----------------------------------------

- Project name "Zacoda" removed everywhere, including from the two new
  scripts and the supervisor. Custom class names (`ZacodaPlusForCausalLM`,
  `ZacodaMTPModule`, `modeling_zacoda_plus.py`) were replaced with generic
  placeholders (`modeling_custom.py`, `CustomForCausalLM`) in comments and
  the one `auto_map` config line that references a modeling file by name.
- All ModelScope-specific details removed: no box hostnames, no account
  details, no mention of the ModelScope platform's session-reset behavior,
  no `/mnt/workspace` or `/dev/shm/cpt_cache` paths (genericized to relative
  paths like `./cpt_cache/cache.jsonl`).
- All Tailscale / Cloudflare Tunnel access details removed.
- No HF_TOKEN, API keys, or secrets.json contents anywhere — checked
  explicitly across all four new/edited files.
- No real name, email, or account handles included. Training-lesson
  anecdotes are written in passive/third-person voice, same as before.
- Absolute local Mac paths (`/Users/SZhang/...`) removed from every file,
  replaced with generic relative paths.
- **The GCS-checkpointing claim was corrected, not just omitted.** The
  original `windowed_sft_cuda.py` docstring described a GCS-checkpointing
  contract (`gsutil cp -r` up/down for spot-preemption resume) as part of its
  design. That's real code that exists in the source file, but it is NOT how
  this pipeline actually runs on the box this repo describes — that box
  checkpoints to local disk, and a separate background process rsyncs to
  network storage on its own schedule, entirely decoupled from the training
  loop. `train_cpt.py` in this repo reflects the REAL design (local atomic
  writes only, no GCS calls anywhere in the file) and the README explicitly
  calls out the discrepancy with the original docstring rather than quietly
  repeating an aspirational claim as fact — same discipline as the earlier
  FlashAttention-2 catch.
- `add_mtp_modules()`, the `--stream`/`PrefetchBuffer` live-HF-streaming path,
  and QAT (`apply_qat`/`_make_qat_linear`) were left OUT of `train_cpt.py`
  and `expand_model.py` entirely, rather than included in a broken or stubbed
  form — all three depend on project-internal modules not in this repo
  (`cpt_streaming_data.py`, a custom MTP module class). Including a stub that
  imports a module that doesn't exist would make the script fail on
  `--stream` or `--mtp-depths`-with-real-MTP-weights in a confusing way; the
  flags that remain (`--cpt-cache`, the `--mtp-depths` config-field-only
  path) work standalone.

THINGS I WASN'T FULLY SURE WERE SAFE — PLEASE DOUBLE-CHECK
-------------------------------------------------------------

- I did NOT include a claim about FlashAttention-2 being used in this
  pipeline (same conclusion as the prior draft — no evidence it's wired into
  the training script, only appears in docs about a different, unrelated
  context).
- The base model family is a third-party "abliterated" Gemma-4 checkpoint
  (from `huihui-ai` on Hugging Face — a public, pre-existing community
  fine-tune, not something built in this project). Still not named in the
  public README, same reasoning as before — add it back deliberately if you
  want the community to know the exact base model.
- All numbers in the README (batch/seqlen OOM comparison, the ~110-iteration
  OOM detail, the 14.7B-param figure, the transformers 5.7.0 version, the
  bitsandbytes reinstall story) came from direct greps of
  `docs/pipeline_state.md`. Please skim the README once yourself to confirm
  nothing reads as more precise/impressive than the source material
  actually supports.
- `catch_and_resume.sh`'s example hyperparameters (iters, batch, lr, paths)
  are illustrative placeholders, not the exact real run configuration — swap
  them for your actual values before using it for a real job.
- No LICENSE file was added — pick one before treating this as reusable.

DISCORD PITCH DRAFT (paste into AMD ROCm Developer Hub "Show and Tell" / "Projects")
---------------------------------------------------------------------------------------

Hey all — sharing a ROCm project I've been building: a vocabulary-pruning →
model-expansion → continued-pretraining pipeline for Gemma-4-family models,
running end-to-end on a single AMD MI300X.

The idea: take a 12B-parameter Gemma-4 base checkpoint, prune out vocabulary
you don't need (script-based heuristic — drops CJK/Cyrillic/Arabic/Devanagari/
Mongolian tokens plus a set of Romance/Germanic diacritic tokens, while
protecting every special/added token), slice the embedding matrix down to
match, expand the model's width and depth to grow its capacity back up
(orthogonal-QR width init, zero-init depth duplication so new layers start as
identity no-ops), and then run continued pretraining on the result — all on
one GPU under ROCm/PyTorch.

A few things I had to actually solve along the way that might be useful to
other people on MI300X:
- bitsandbytes 8-bit Adam vs plain AdamW: losing bitsandbytes on a fresh
  container silently falls back to fp32 AdamW, which needs ~4x the optimizer
  memory — that was enough to OOM a 14.7B-parameter run about 110 iterations
  in, once the optimizer's state had fully allocated. Now there's an explicit
  reinstall step and a resume-time guard that refuses to load one optimizer
  type's state into a different one (this used to silently corrupt memory
  usage instead of erroring cleanly).
- Async checkpointing to LOCAL disk: GPU→CPU snapshot synchronously, then the
  actual disk write happens on a background thread so training doesn't stall
  for the full write — atomic temp-dir-then-rename so a killed process can
  never leave a half-written checkpoint at the real path. No cloud object
  store in the loop at all; getting checkpoints onto durable storage is a
  separate, decoupled rsync step.
- Local JSONL data caching instead of live HF streaming, so a flaky network
  path to the data source doesn't stall the GPU — trains with zero network
  calls once the cache exists.
- A concrete, measured batch/seqlen tradeoff: batch=2 at seqlen=1024 uses
  less memory than batch=1 at seqlen=2048 for the same tokens/step, because
  attention scales ~O(seq_len²) — moving to the smaller-seqlen/larger-batch
  config turned two repeated OOMs into a stable run.
- A crash-recovery supervisor script that relaunches training automatically,
  keeping a loss-tagged checkpoint history so a bad training patch can be
  rolled back instead of silently becoming the only checkpoint left.

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

1. Review every file in this directory yourself — especially the four
   "wasn't fully sure" items above.
2. Decide on a license and add a LICENSE file if you want this to be
   genuinely reusable.
3. Create the actual GitHub repo yourself (this task did not touch
   github.com in any way) and push this local repo to it yourself.
4. Post to Discord / any forum yourself, using the draft above as a
   starting point — no message was sent anywhere by this task.
