# Optimization Plan — Phase G

Based on a full-codebase audit, here are the remaining optimization opportunities prioritized by impact/effort. This plan covers the top 10 items; items 12-18 are lower priority and deferred.

---

## Step 1: NCCL topology env auto-set (easy, high multi-GPU impact)
**File**: `rocm_env.py` or new `nccl_env.py`
- Detect interconnect via `rocm-smi --showtopo` / `lspci`
- Auto-set `NCCL_SOCKET_IFNAME`, `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE` when appropriate
- Log what was set; keep user-overridable
- Currently only `NCCL_ASYNC_ERROR_HANDLING` + `NCCL_DEBUG` are set (`train_cpt.py:1245-1246`)

## Step 2: hipGraph capture for inference decode (medium, high latency impact)
**File**: `generate.py`
- `--static-cache` already sets `cache_implementation="static"` but never captures a CUDA/HIP graph
- After warmup, wrap single-step decode in `torch.cuda.make_graphed_callables` or document `torch.compile(mode="reduce-overhead")` as the decode path
- 1.5-3x decode speedup on MI300X; scaffolding is already in place

## Step 3: FP8 scaling config tuning (medium, high impact — MI300X's core feature)
**File**: `train_cpt.py` `_apply_fp8`
- Currently calls `convert_to_float8_training(model)` with NO config (torchao defaults = e4m3fn for everything, per-step dynamic scaling)
- Add `Float8LinearConfig` with: e5m2 for backward gradients (wider range), delayed/historical scaling with amax history window (16 steps) for stability
- This is the standard production FP8 recipe

## Step 4: vLLM/SGLang serving backend (hard, high serving throughput impact)
**File**: `serve.py`
- Add optional `--backend vllm` path that loads via `vllm.LLM` and routes through its continuous-batching scheduler
- Keep current `transformers.generate` as `--backend hf` default
- This is the single biggest inference-throughput gap vs production AMD deployment

## Step 5: DCP sharded checkpointing (medium, removes multi-GPU stall)
**File**: `train_cpt.py` + `async_checkpoint.py`
- Currently FSDP checkpoint does a full state-dict gather (blocking all-gather) then async-writes only the disk I/O
- Switch to `torch.distributed.checkpoint` with `ShardedStateDictConfig` — each rank writes its own shard, no full gather
- Single-GPU mode unaffected

## Step 6: pyproject.toml extras + Dockerfile improvements (easy, cheap wins)
**Files**: `pyproject.toml`, `Dockerfile`
- Add `[serve]`, `[train]`, `[infer]`, `[all]` extras to pyproject.toml
- Pin Dockerfile base image to a real verified ROCm tag (currently `:latest`)
- Add multi-stage build (builder compiles flash-attn, runtime copies wheel)

## Step 7: Metrics sink abstraction + evaluate.py flags (easy, medium impact)
**Files**: `train_cpt.py`, `evaluate.py`
- Add `JsonlSink` (zero-dep JSON metrics per step) alongside TensorBoard
- Add `--flash-attn`, `--fp8`, `--compile` flags to `evaluate.py` so eval matches train/inference config

## Step 8: fp32 master-weight option + torch LR scheduler (easy, medium impact)
**Files**: `train_cpt.py`
- Add `--master-fp32` flag: load model in fp32, FSDP `MixedPrecision` casts to bf16 for compute, keeps fp32 master weights for long CPT stability
- Replace hand-rolled `lr_at_step` with `LambdaLR` (preserves resume-offset via `last_epoch`)

## Step 9: ROCm CI job + benchmark regression (medium, confidence impact)
**Files**: `.github/workflows/`, `results/baseline.json`
- Add `rocm-selftest` job on self-hosted MI300X (10-iter smoke training + short generate)
- Add benchmark-regression job asserting throughput within X% of baseline

## Step 10: Fused linear+cross-entropy (medium, high for large-vocab/MTP)
**Files**: `modeling_custom.py`, `train_cpt.py`
- For 256k-vocab models, `lm_head` materializes ~2GB of logits per forward (and MTP calls it per depth)
- Use Liger Kernel or a fused linear+CE op to compute loss without materializing logits
- Saves multi-GB peak memory + bandwidth

---

## Suggested first 3 PRs
1. **"NCCL env + pyproject extras + Dockerfile pin"** — Steps 1, 6 (easy, high ROI)
2. **"hipGraph decode + FP8 scaling config + evaluate.py flags"** — Steps 2, 3, 7 (medium, MI300X headline features)
3. **"DCP sharded checkpointing + vLLM serving backend"** — Steps 4, 5 (hard, production deployment)