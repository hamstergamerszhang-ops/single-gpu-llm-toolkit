#!/usr/bin/env python3
"""Standalone non-blocking checkpoint writer for single-GPU training loops.

Extracted from train_cpt.py's AsyncCheckpointer class (and its module-level
_move_to_cpu helper) into its own file so the pattern is importable/reusable
without the whole training script, and so its two-phase design is documented
in one place instead of buried inside the training script's main().

The pattern (same as train_cpt.py's synchronous atomic_save_checkpoint, but
split into a sync phase and an async phase):
  1. SYNCHRONOUS snapshot -- copy model + optimizer state to CPU RAM. This
     still blocks the training loop briefly (a GPU->CPU copy over
     PCIe/interconnect), but it's a small fraction of the time a full disk
     write takes on a slow mount, and it's the only phase that MUST be
     synchronous: the GPU tensors are about to be mutated by the next
     training step, so the copy has to happen before that step runs.
  2. ASYNC write -- everything from "turn these CPU tensors into files on
     disk" onward runs on a background thread and never touches the live
     model/optimizer again, so it's safe to run concurrently with the next
     several training steps. Uses the same atomic write pattern as the
     synchronous path: write to a `.tmp_ckpt` dir, then os.replace() onto
     the real path, so a kill -9 mid-write never leaves a corrupted
     checkpoint observable at the real path.

Bounded to ONE in-flight write at a time -- save() blocks on any
still-running previous write before starting a new snapshot. This prevents
unbounded CPU-RAM growth from queueing multiple large snapshots if writes
fall behind the checkpoint interval, at the cost of occasionally still
waiting on a slow write. With a sane checkpoint interval relative to disk
speed this should be rare in practice.

This class only ever writes to LOCAL disk. Getting checkpoints onto
durable/shared storage (e.g. a periodic rsync to a network volume) is a
separate, deliberately decoupled concern -- this module does not wire in
any cloud-object-store push.

Usage as a library:
    from async_checkpoint import AsyncCheckpointer
    ckpt = AsyncCheckpointer()
    ...
    ckpt.save(model, optimizer, step, save_dir, tokenizer=tokenizer)
    ...
    ckpt.wait()   # call before process exit (SIGTERM handler, final checkpoint)

Self-test (no GPU/model required -- uses a tiny torch.nn.Module + real
threads + a tmp dir, exercises the full snapshot -> background-write ->
atomic-rename path end to end):
    python3 async_checkpoint.py --selftest
"""

import argparse
import os
import shutil
import threading
from pathlib import Path


def _move_to_cpu(obj):
    """Recursively moves tensors in a nested dict/list to CPU. Used to snapshot
    optimizer.state_dict() (a dict of dicts of tensors, e.g. Adam's per-param
    moment buffers) before handing it to a background thread -- the
    GPU-resident originals keep getting mutated by the next training step the
    moment this snapshot is taken, so nothing downstream may still reference
    the live GPU tensors."""
    import torch
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    return obj


class AsyncCheckpointer:
    """Non-blocking checkpoint writes: the expensive part (serializing tens of
    GB to a possibly-slow disk/NFS mount) happens on a background thread
    while the GPU immediately continues training the next step, instead of
    sitting idle for the whole write. See module docstring for the two-phase
    design and why only ONE write is ever in flight at a time.
    """

    def __init__(self):
        self._thread = None
        # If a background write raised, the exception is stored here and
        # re-raised on the next save()/wait_for_pending() call. Without this,
        # a failed write (disk full, NFS error, permission denied) would die
        # silently inside the thread — the training loop would keep running in
        # the belief that it has checkpoints it does not have.
        self._last_error = None

    def is_write_in_flight(self) -> bool:
        """True if a background write is currently running. save() will
        block on this before starting a new snapshot; callers that want to
        avoid blocking can check this first and skip a checkpoint tick."""
        return self._thread is not None and self._thread.is_alive()

    def check_error(self):
        """Re-raise any exception captured by the background write thread.
        Call this after save() (or at any point) to surface a failed write
        instead of continuing to train checkpoint-less."""
        if self._last_error is not None:
            err = self._last_error
            self._last_error = None
            raise RuntimeError(
                f"async checkpoint write failed (training continued without a "
                f"valid checkpoint on disk): {err}"
            ) from err

    def save(self, model, optimizer, step: int, save_dir: Path, tokenizer=None,
             extra_state: dict | None = None, custom_code_src: Path | None = None):
        # Surface any error from a PRIOR background write before starting a new
        # one — otherwise we'd silently overwrite the lost-write state with a
        # fresh snapshot and the prior failure would be gone forever.
        self.check_error()
        if self.is_write_in_flight():
            print("[ckpt-async] previous async write still in flight -- waiting for it "
                  "before starting a new snapshot (checkpoint interval may be too tight "
                  "relative to write speed on this disk)")
            self._thread.join()

        # Phase 1 (synchronous, blocks briefly): snapshot to CPU.
        # Drop tied-weight duplicates BEFORE copying (e.g. tie_word_embeddings=True
        # means lm_head.weight and embed_tokens.weight share the same storage) --
        # a naive model.state_dict() copy loses the tie relationship (each
        # .to('cpu', copy=True) makes an independent copy), and passing that as an
        # explicit state_dict= override to save_pretrained() skips its normal
        # tied-weight dedup, silently writing a full extra copy of the embedding
        # matrix (GB-scale on a large-vocab model) into every checkpoint. Removing
        # the known-duplicate key here restores the synchronous path's behavior
        # exactly -- transformers re-derives it from config on load either way.
        tied_keys = set(getattr(model, "_tied_weights_keys", {}) or {})
        raw_state = model.state_dict()
        model_state_cpu = {k: v.detach().to("cpu", copy=True)
                           for k, v in raw_state.items() if k not in tied_keys}
        opt_state_cpu = {
            "optimizer": _move_to_cpu(optimizer.state_dict()),
            "optimizer_type": type(optimizer).__name__,
            "step": step,
            **(extra_state or {}),
        }
        print(f"[ckpt-async] step {step}: CPU snapshot done, disk write continuing in "
              f"background -- training resumes immediately")

        save_dir = Path(save_dir)

        # Phase 2 (async): everything below only touches CPU tensors/disk, never the
        # live model/optimizer, so it's safe to run while training proceeds.
        # Errors here are captured into self._last_error rather than propagating
        # (an uncaught exception in a thread just prints a traceback to stderr
        # and dies — the training loop would never know the write failed). The
        # next save()/wait_for_pending() call re-raises it via check_error().
        def _write():
            import torch
            try:
                tmp_dir = save_dir.parent / (save_dir.name + ".tmp_ckpt")
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(tmp_dir, safe_serialization=True, state_dict=model_state_cpu)
                if tokenizer is not None:
                    tokenizer.save_pretrained(tmp_dir)
                if custom_code_src is not None:
                    src_file = Path(custom_code_src) / "modeling_custom.py"
                    if src_file.exists():
                        shutil.copy2(src_file, tmp_dir / "modeling_custom.py")

                torch.save(opt_state_cpu, tmp_dir / "training_state.pt")

                # Retain the previous checkpoint as .prev (a real backup, not
                # deleted) so a crash mid-write or a corrupt new write can be
                # rolled back. The recovery path in train_cpt.py restores .prev
                # if the live save_dir is missing training_state.pt on resume.
                backup = save_dir.parent / (save_dir.name + ".prev")
                if save_dir.exists():
                    if backup.exists():
                        shutil.rmtree(backup)
                    os.replace(save_dir, backup)
                os.replace(tmp_dir, save_dir)
                # NOTE: .prev is intentionally NOT deleted here — it is the
                # retained backup. The next successful write rotates it out
                # (rmtree + os.replace above) when a newer good checkpoint
                # supersedes it.

                print(f"[ckpt-async] step {step}: background write finished -> {save_dir}")
            except Exception as e:
                self._last_error = e
                print(f"[ckpt-async] step {step}: BACKGROUND WRITE FAILED — "
                      f"error stored, will be raised on next save()/wait. "
                      f"Previous checkpoint retained at {save_dir}.prev if it "
                      f"existed. Failure: {e}", file=__import__('sys').stderr)

        self._thread = threading.Thread(target=_write, daemon=False)
        self._thread.start()

    def wait_for_pending(self):
        """Block until any in-flight write finishes -- call before process exit
        (SIGTERM handler, final checkpoint) so the process never dies mid-write.
        Re-raises any error captured by the background thread."""
        if self.is_write_in_flight():
            print("[ckpt-async] waiting for final background write to finish before exit...")
            self._thread.join()
        self.check_error()


def _self_test():
    import tempfile

    print("[selftest] AsyncCheckpointer: end-to-end snapshot -> background write -> "
          "atomic rename, against a tiny real torch.nn.Module (no GPU needed)")
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.fc(x)

        def save_pretrained(self, out_dir, safe_serialization=True, state_dict=None):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            sd = state_dict if state_dict is not None else self.state_dict()
            torch.save(sd, out_dir / "model_state.pt")

    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Take one real step so the optimizer has non-empty state to snapshot.
    loss = model(torch.randn(1, 4)).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "ckpt"
        ckpt = AsyncCheckpointer()
        assert not ckpt.is_write_in_flight()
        ckpt.save(model, optimizer, step=1, save_dir=save_dir)
        ckpt.wait_for_pending()
        assert not ckpt.is_write_in_flight()
        assert (save_dir / "model_state.pt").exists()
        assert (save_dir / "training_state.pt").exists()
        state = torch.load(save_dir / "training_state.pt", weights_only=False)
        assert state["step"] == 1
        assert state["optimizer_type"] == "AdamW"
        assert not (save_dir.parent / (save_dir.name + ".tmp_ckpt")).exists()
        # First write: no prior checkpoint, so .prev is not created.
        assert not (save_dir.parent / (save_dir.name + ".prev")).exists()
        print("  OK (model + optimizer state on disk, no leftover tmp/backup dirs)")

        print("[selftest] a second save() overwrites cleanly and preserves the new step")
        loss = model(torch.randn(1, 4)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ckpt.save(model, optimizer, step=2, save_dir=save_dir)
        ckpt.wait_for_pending()
        state = torch.load(save_dir / "training_state.pt", weights_only=False)
        assert state["step"] == 2
        # Second write: the prior (step 1) checkpoint is RETAINED as .prev, not
        # deleted — so a crash or corrupt later write can roll back to it.
        prev = save_dir.parent / (save_dir.name + ".prev")
        assert prev.exists(), "second write should retain .prev backup"
        prev_state = torch.load(prev / "training_state.pt", weights_only=False)
        assert prev_state["step"] == 1, ".prev should hold the prior (step 1) checkpoint"
        print("  OK (step 2 live; step 1 retained as .prev backup)")

        print("[selftest] a failed background write surfaces its error on the next call")
        # Monkeypatch save_pretrained to raise, forcing the background thread to
        # capture the error. Without check_error()/self._last_error, this error
        # would be silently swallowed and training would continue checkpoint-less.
        def raise_save_pretrained(*a, **kw):
            raise OSError("simulated disk full")
        model.save_pretrained = raise_save_pretrained
        ckpt.save(model, optimizer, step=3, save_dir=save_dir)
        # wait_for_pending() re-raises via check_error() — the error surfaces
        # here, exactly where a caller would discover it on exit.
        raised = False
        try:
            ckpt.wait_for_pending()
        except RuntimeError as e:
            raised = True
            assert "simulated disk full" in str(e), str(e)
        assert raised, "wait_for_pending() should re-raise the background write failure"

        # A second failing write: save() returns (the write runs in the
        # background), and the error surfaces on the next wait_for_pending().
        # This is the correct semantics — save() surfaces PRIOR errors (via
        # check_error at its top), not the current write's error (which hasn't
        # happened yet when save() returns).
        ckpt.save(model, optimizer, step=4, save_dir=save_dir)
        raised2 = False
        try:
            ckpt.wait_for_pending()
        except RuntimeError as e:
            raised2 = True
            assert "simulated disk full" in str(e), str(e)
        assert raised2, "second failed write should surface on wait_for_pending()"
        print("  OK (failed write errors surface via wait_for_pending(), not swallowed)")



    print("[selftest] _move_to_cpu recurses through nested dict/list of tensors")
    nested = {"a": torch.randn(2, 2), "b": [torch.randn(1), {"c": torch.randn(3)}]}
    moved = _move_to_cpu(nested)
    assert moved["a"].device.type == "cpu"
    assert moved["b"][0].device.type == "cpu"
    assert moved["b"][1]["c"].device.type == "cpu"
    print("  OK")

    print("\n[selftest] All checks passed (no GPU required -- run a real checkpoint "
          "against your actual model/optimizer on hardware before trusting this for a "
          "real training job).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    args = ap.parse_args()
    if args.selftest:
        _self_test()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
