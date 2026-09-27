# Real MI300X training run — Zacoda Lite E4B continued-pretraining

This documents a real, completed continued-pretraining (CPT) run of a ~4B parameter model
(Gemma-4 E4B architecture) on a rented single AMD Instinct MI300X GPU, using the tools in
this repository. Every number below is either a directly observed value from the live
training log at the time, or explicitly marked as an estimate.

## What was run

- **Model**: ~4B params (Gemma-4 E4B architecture), continuing from a vocab-pruned,
  previously-CPT'd checkpoint (this was the 4th pass in a v2→v5 curated-data progression;
  v5 is the final and largest pass).
- **Hardware**: 1x AMD Instinct MI300X (192GB HBM3), rented cloud instance, ROCm PyTorch.
- **Training script**: `train_cpt.py`-equivalent in this repo (`windowed_sft_cuda.py --cpt`
  in the source project) — raw-text continued-pretraining, no prompt masking.
- **Data**: 1,586,966 rows, a quality-filtered and category-balanced curated mix (general
  18% / logic+reasoning 35% / math 8% / code 16% / agentic tool-use 17% / identity+anti-hack
  1% / voice 3%), including 10,000 originally-authored examples.
- **Command**:
  ```
  python3 windowed_sft_cuda.py --model <prev_checkpoint> --data <data_dir> --save <out_dir> \
    --cpt --batch 16 --iters 3000 --lr 5e-6 --checkpoint-every 200 --async-checkpoint
  ```

## Real measured numbers

| Metric | Value | Source |
|---|---|---|
| Optimizer | bitsandbytes 8-bit Adam | confirmed in training log at launch |
| Batch size (stable) | 16 | proven across the full run; batch 20 was tried once and OOM'd at iter ~90 (see "What broke" below) |
| Peak VRAM at batch 16 | ~185–202 GiB of 191.69 GiB total reported capacity (~97–99%) | multiple direct `rocm-smi --showmeminfo vram` reads during the run |
| Sustained power draw | 360–460 W | multiple direct `rocm-smi --showpower` reads during the run |
| GPU utilization | 100% (sustained) | multiple direct `rocm-smi --showuse` reads during the run |
| Training iterations completed | 3000 / 3000 (target reached) | confirmed by direct inspection of the final checkpoint's saved state (`step: 3000`), read via a partial remote ZIP fetch of the checkpoint file's internal pickle stream — see "Verification method" below |

**Real loss values directly observed in the live training log** (not interpolated or estimated):

| Iteration | Loss |
|---|---|
| 1200 | 0.9485 |
| 1400 | 0.8010 |
| 1800 | 1.1300 |
| 2000 | 1.1043 |
| 2200 | 0.7036 |
| 2400 | 1.2487 |
| 2600 | 1.0148 |

Loss oscillates in the 0.6–1.4 range throughout — expected and healthy for CPT on a
diverse, category-mixed corpus (unlike SFT on a narrow task, CPT loss does not monotonically
decrease toward zero; the metric to watch is stability, not the raw number). **Honest gap**:
the exact loss value printed at iteration 3000 itself was not captured before the training
box was reclaimed — the run's completion (reaching the full 3000-iteration target) is
independently confirmed via the checkpoint file itself, just not paired with that specific
log line. No fabricated number is given here.

## What broke, and what survived it

This run is presented with its real operational history included, not a clean-room retelling:

1. **OOM at iteration ~90** — an attempt to use batch size 20 (rather than the
   proven-stable 16) ran out of memory (`torch.OutOfMemoryError`, tried to allocate 29 GiB
   with 3.39 GiB free). No checkpoint had been written yet at that point (checkpoints save
   every 200 iterations), so nothing was lost — the run was relaunched at batch 16 and never
   OOM'd again.
2. **A full rented-instance failure and recovery.** The cloud instance running this job was
   reclaimed mid-run. Training resumed on a fresh instance from the last checkpoint
   successfully written before the failure (iteration 1600) — both from a hub-hosted backup
   and, separately, from the account's persistent network storage, which is confirmed to
   survive full instance replacement, not just instance restarts.
3. **A persistent-storage quota exhaustion mid-run**, which broke the local backup mirror
   (not the primary hub backup) for one cycle. Freed by removing a superseded, already
   separately-backed-up earlier checkpoint. Training itself was never interrupted by this —
   it only affected a secondary backup path.
4. **A second instance reclaim, this time after the run had already reached its full
   3000-iteration target** and pushed its final checkpoint out successfully — confirmed
   after the fact (see "Verification method" below), so no relaunch was needed.

Across all of this, **zero training progress was permanently lost** — the worst-case loss at
any single point was the gap between a 200-iteration checkpoint interval and whenever the
next backup cycle ran, never more.

## Verification method: confirming completion without a live GPU

After the second instance reclaim, there was no running process or live log to read from
directly. Completion was confirmed by fetching just the small internal metadata entry
(`data.pkl`, a few hundred KB) out of the ~8 GB checkpoint file using HTTP range requests
against the hub-hosted backup — reading a PyTorch checkpoint's ZIP-format internal structure
directly, without downloading the full file, and without needing GPU access at all. The
pickle's disassembled opcode stream showed `step: 3000` unambiguously. This is a genuinely
useful, reusable pattern for verifying checkpoint state remotely and cheaply, and the exact
same technique is what confirmed the final result documented here.

## What this demonstrates

- Real, full-parameter (no LoRA) continued-pretraining of a multi-billion-parameter model
  runs correctly on a single AMD MI300X via ROCm PyTorch, for the full duration of a real,
  non-trivial training job (not a toy/smoke-test scale).
- The tooling in this repository survives the realistic failure modes of rented single-GPU
  cloud compute — OOM, instance loss, storage limits — without silent data loss, using
  nothing beyond a checkpoint-and-backup discipline that's practical for an individual
  developer to run, not just a well-resourced lab.

---

Questions or want the raw curated dataset spec / exact tooling used: hamstergamerszhang@gmail.com
