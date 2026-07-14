#!/usr/bin/env python3
"""Offline pre-tokenizer: turns JSONL training data into pre-tokenized .pt
shards so train_cpt.py can skip per-step tokenization entirely.

WHY THIS EXISTS
---------------
In train_cpt.py the data path calls build_cpt_example / build_sft_example on
every row inside the training loop (or just before it), which means the HF
tokenizer's Python/Regex frontend runs once per example per epoch. On a
single-GPU ROCm box that GPU is almost always the bottleneck of the FORWARD
and BACKWARD passes, but the tokenizer runs on the CPU and serially blocks
the next batch from being prepared. On long CPT runs (millions of rows) the
tokenizer's wall-clock is a real fraction of total step time, and it is pure
waste on epoch 2+ because the token ids never change. Pre-tokenizing once,
into sharded .pt files of ready-to-stack `input_ids` / `labels` tensors, lets
the trainer mmap a shard, index a row, and call .to(device) — no tokenizer in
the hot path at all. This is the single biggest throughput win available on a
single-GPU box that can't overlap a DataLoader worker pool against the GPU
without eating VRAM/CPU the trainer also wants.

WHAT IT PRODUCES
----------------
Each shard file is a torch.save'd dict:

    {
      "input_ids": [Tensor[long], ...],   # one 1-D tensor per row, variable length
      "labels":    [Tensor[long], ...],   # same shape as input_ids per row
      "meta": {"mode": "cpt"|"sft", "shard_idx": N},
    }

Rows are NOT padded to a uniform length and are NOT sequence-packed (no
concatenation across rows) — padding/packing stays the trainer's job at
collate time, so this cache is independent of --batch / --max-seq-len choices
up to the per-row truncation length used at pre-tokenize time (see
--max-seq-len, which must be >= the trainer's --max-seq-len or rows will have
been truncated shorter than the trainer expects; rows are never longer than
--max-seq-len).

MASKING (SFT mode)
------------------
For {"messages":[...]} rows, assistant-turn-only labels are produced using the
EXACT same chat-template masking logic as train_cpt.py's build_sft_example:
incremental apply_chat_template per turn, diff against the running prefix,
label only assistant spans, -100 on everything else. The two functions below
(build_sft_example / build_cpt_example) are a VERBATIM copy of the ones in
train_cpt.py — they MUST be kept in sync if the masking there ever changes.
See build_sft_example's docstring for the appenditive-template assumption and
its non-appenditive fallback (both carried over unchanged).

HONEST CAVEATS
--------------
  - Tokenization is CPU-only; the GPU is not used. We still call
    rocm_env.setup_rocm_env() before importing torch, for two reasons: (1) it
    is a no-op on non-ROCm boxes so it costs nothing, and (2) we DO import
    torch to build/save tensors, and on a ROCm box where the wheel needs an
    HSA_OVERRIDE_GFX_VERSION the override must be set before torch's runtime
    initializes — even though we only ever place tensors on CPU. The env
    bootstrap does not speed up tokenization; it just keeps this tool from
    being the thing that trips a "no kernel image" error on a misdetected box.
  - The cache is keyed by NOTHING automatic: if you change --tokenizer,
    --mode, --max-seq-len, or the chat template, you must re-run. There is no
    fingerprint check here — the trainer is responsible for pointing at a
    cache that matches its config.
  - Memory: rows are tokenized streaming and only `--shard-size` rows are held
    in memory at once, so this scales to arbitrarily large JSONL without
    loading it all. Each shard is written and the buffer cleared before the
    next is filled.

Usage:
    python3 pretokenize.py \\
        --src data.jsonl --tokenizer ./checkpoints/base \\
        --dst ./cpt_cache --mode cpt --shard-size 10000

    # SFT (assistant-turn masking):
    python3 pretokenize.py \\
        --src sft.jsonl --tokenizer ./checkpoints/base \\
        --dst ./sft_cache --mode sft --shard-size 5000 --max-seq-len 4096

Self-test (no GPU / no tokenizer / no torch required for the core checks —
exercises JSONL streaming, shard grouping, and shard-dict assembly with fake
data; a guarded torch round-trip + masking check runs only if torch is
importable):
    python3 pretokenize.py --selftest
"""

import argparse
import json
import sys
from pathlib import Path


def log(msg: str):
    print(f"[pretokenize] {msg}", flush=True)


# ── JSONL streaming ──────────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    """Yield parsed dicts from a JSONL file, streaming (constant memory).
    Blank lines are skipped; malformed JSON lines are skipped with a warning
    on stderr rather than aborting the whole run — matches train_cpt.py's
    load_jsonl tolerance, but streaming instead of materializing the list."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[pretokenize] WARNING: skipping malformed JSON at "
                      f"{path}:{lineno}: {e}", file=sys.stderr)


# ── example builders (imported from shared tokenization.py) ─────────────────
#
# Previously these were verbatim copies of train_cpt.py's build_sft_example /
# build_cpt_example — a silent-divergence risk. Now imported from the shared
# tokenization.py module so there is ONE source of truth for the masking logic.
from tokenization import build_sft_example, build_cpt_example


def tokenize_row(row: dict, tokenizer, mode: str, max_seq_len: int):
    """Dispatch to the CPT or SFT builder. Returns the {"input_ids","labels"}
    dict, or None if the row doesn't have the field its mode needs (logged +
    skipped rather than crashing the whole run)."""
    if mode == "cpt":
        if "text" not in row:
            log(f"WARNING: cpt row missing 'text' field — skipping")
            return None
        return build_cpt_example(row, tokenizer, max_seq_len)
    elif mode == "sft":
        if "messages" not in row:
            log(f"WARNING: sft row missing 'messages' field — skipping")
            return None
        return build_sft_example(row, tokenizer, max_seq_len)
    raise ValueError(f"unknown mode {mode!r} (expected 'cpt' or 'sft')")


# ── shard grouping / assembly (pure-python, no torch — testable standalone) ──

def group_into_shards(items, shard_size: int):
    """Yield lists of at most `shard_size` items from an iterable. The final
    list may be shorter. Pure-python so --selftest can exercise it without
    torch. Raises ValueError for shard_size <= 0 (a 0/empty shard would
    silently drop every row, which is almost certainly a typo)."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be >= 1 (got {shard_size})")
    buf = []
    for it in items:
        buf.append(it)
        if len(buf) >= shard_size:
            yield buf
            buf = []
    if buf:
        yield buf


def assemble_shard(examples: list[dict], mode: str, shard_idx: int) -> dict:
    """Build the shard dict {"input_ids":[...], "labels":[...], "meta":{...}}
    from a list of per-row example dicts. Pure-python: it just collects the
    tensors/lists each example already carries, so it works with real torch
    tensors in production AND with plain lists in --selftest (which is why the
    type is generic, not Tensor)."""
    return {
        "input_ids": [ex["input_ids"] for ex in examples],
        "labels": [ex["labels"] for ex in examples],
        "meta": {"mode": mode, "shard_idx": shard_idx},
    }


def save_shard(shard: dict, dst_dir: Path, shard_idx: int):
    """torch.save the shard dict to <dst_dir>/shard_<NNNNNN>.pt. Zero-padded
    so an `ls`/glob sorts lexicographically = numerically."""
    import torch

    path = dst_dir / f"shard_{shard_idx:06d}.pt"
    torch.save(shard, path)
    return path


# ── main pipeline ────────────────────────────────────────────────────────────

def run_pretokenize(src: Path, tokenizer_path: str, dst: Path, mode: str,
                    shard_size: int, max_seq_len: int,
                    gfx_override: str | None, hip_alloc_conf: str | None):
    """Stream src, tokenize each row, group into shards of `shard_size`, and
    write each shard as a .pt file under dst. Returns (n_rows, n_tokens,
    n_shards)."""
    # ROCm env bootstrap: MUST run before `import torch`. No-op on non-ROCm
    # boxes; on a ROCm box whose wheel needs HSA_OVERRIDE_GFX_VERSION this
    # prevents a "no kernel image" error at torch init even though we only
    # place tensors on CPU. See module docstring for the honesty note on why
    # this is here despite tokenization being CPU-only.
    from rocm_env import setup_rocm_env
    setup_rocm_env(override=gfx_override, hip_alloc_conf=hip_alloc_conf)

    import torch  # noqa: F401  (env must be set first; torch used below)
    from transformers import AutoTokenizer

    log(f"loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    dst.mkdir(parents=True, exist_ok=True)
    log(f"mode={mode} shard_size={shard_size} max_seq_len={max_seq_len} -> {dst}")

    shard_buf: list[dict] = []
    shard_idx = 0
    n_rows = 0
    n_tokens = 0
    n_skipped = 0

    for row in iter_jsonl(src):
        try:
            ex = tokenize_row(row, tokenizer, mode, max_seq_len)
        except Exception as e:  # tokenizer frontend can raise on odd unicode/etc.
            log(f"WARNING: tokenization failed for a row ({type(e).__name__}: {e}) "
                f"— skipping")
            n_skipped += 1
            continue
        if ex is None:
            n_skipped += 1
            continue
        shard_buf.append(ex)
        n_rows += 1
        n_tokens += int(ex["input_ids"].numel())
        if len(shard_buf) >= shard_size:
            shard = assemble_shard(shard_buf, mode, shard_idx)
            path = save_shard(shard, dst, shard_idx)
            log(f"wrote {path.name}: {len(shard_buf)} rows, "
                f"{sum(int(t.numel()) for t in shard['input_ids'])} tokens")
            shard_idx += 1
            shard_buf = []

    if shard_buf:
        shard = assemble_shard(shard_buf, mode, shard_idx)
        path = save_shard(shard, dst, shard_idx)
        log(f"wrote {path.name}: {len(shard_buf)} rows (final partial shard)")
        shard_idx += 1

    log(f"done: {n_rows} rows, {n_tokens} tokens, {shard_idx} shard(s) -> {dst} "
        f"({n_skipped} row(s) skipped)")
    return n_rows, n_tokens, shard_idx


# ── self-test (no GPU / no tokenizer required) ───────────────────────────────

class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer, used ONLY when --selftest opts
    into the guarded torch masking check. apply_chat_template renders an
    appenditive template (each turn appended verbatim, so the prefix
    assumption build_sft_example relies on holds), and __call__ "tokenizes"
    by mapping whitespace-separated words to deterministic ints. Not a real
    tokenizer — just enough surface to exercise build_sft_example's masking
    without pulling in transformers."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        parts = [f"{m['role']}: {m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("assistant: ")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False, truncation=False,
                 max_length=None):
        ids = [(sum(ord(c) for c in w) % 1000) + 1 for w in text.split()]
        return {"input_ids": ids}


def self_test():
    import tempfile

    print("[selftest] pretokenize: JSONL streaming skips blank/malformed lines")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "in.jsonl"
        p.write_text(
            '{"text": "alpha beta"}\n'
            '\n'                                   # blank — skip
            '{"text": "gamma"}\n'
            'not json at all\n'                    # malformed — skip
            '{"text": "delta epsilon zeta"}\n',
            encoding="utf-8",
        )
        rows = list(iter_jsonl(p))
        assert len(rows) == 3, rows
        assert rows[0]["text"] == "alpha beta"
        assert rows[2]["text"] == "delta epsilon zeta"
        print("  OK (3 valid rows kept out of 5 lines; blank + malformed skipped)")

    print("[selftest] shard grouping: full shards + final partial shard")
    shards = list(group_into_shards(range(7), 3))
    assert shards == [[0, 1, 2], [3, 4, 5], [6]], shards
    print("  OK (7 items / shard_size 3 -> [3,3,1])")

    shards = list(group_into_shards(range(6), 3))
    assert shards == [[0, 1, 2], [3, 4, 5]], shards  # no empty trailing shard
    print("  OK (exact multiple -> no empty trailing shard)")

    shards = list(group_into_shards(iter(["a"]), 100))
    assert shards == [["a"]], shards  # single-item final shard still emitted
    print("  OK (single item below shard_size still yields one shard)")

    try:
        list(group_into_shards(range(3), 0))
        raise AssertionError("expected ValueError for shard_size=0")
    except ValueError:
        pass
    print("  OK (shard_size <= 0 raises rather than silently dropping all rows)")

    print("[selftest] shard dict assembly: structure + meta (no torch needed)")
    fake_examples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    shard = assemble_shard(fake_examples, "cpt", 2)
    assert shard["input_ids"] == [[1, 2, 3], [4, 5]], shard["input_ids"]
    assert shard["labels"] == [[1, 2, 3], [-100, 5]], shard["labels"]
    assert shard["meta"] == {"mode": "cpt", "shard_idx": 2}, shard["meta"]
    # input_ids / labels stay aligned per-row (no transpose).
    assert len(shard["input_ids"]) == len(shard["labels"]) == 2
    print("  OK (per-row lists preserved, meta carries mode + shard_idx)")

    # Guarded torch checks: only run if torch is importable on this host. The
    # core selftest above is fully torch-free, so --selftest still passes on a
    # box without torch installed (e.g. a CI runner for the pure-python logic).
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[selftest] torch not installed — skipping save/load round-trip "
              "and SFT masking check (core logic above is torch-free).")
        print("\n[selftest] All core checks passed (no GPU/torch/tokenizer "
              "required).")
        return

    print("[selftest] torch round-trip: save -> load preserves tensors + meta")
    import torch
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        examples = [
            {"input_ids": torch.tensor([10, 20, 30], dtype=torch.long),
             "labels": torch.tensor([10, 20, 30], dtype=torch.long)},
            {"input_ids": torch.tensor([40, 50], dtype=torch.long),
             "labels": torch.tensor([-100, 50], dtype=torch.long)},
        ]
        shard = assemble_shard(examples, "sft", 0)
        path = save_shard(shard, td, 0)
        loaded = torch.load(path, weights_only=False)
        assert loaded["meta"] == {"mode": "sft", "shard_idx": 0}
        assert len(loaded["input_ids"]) == 2
        assert torch.equal(loaded["input_ids"][0], examples[0]["input_ids"])
        assert torch.equal(loaded["labels"][1], examples[1]["labels"])
        print("  OK (loaded tensors equal saved; meta intact)")

    print("[selftest] SFT masking: assistant spans labeled, prompt spans -100")
    # Fake appenditive tokenizer -> build_sft_example's primary (non-fallback)
    # path runs. Two assistant turns, two user turns: we assert (a) labels and
    # input_ids have equal length, (b) at least some tokens are labeled (the
    # assistant output), and (c) at least some are -100 (the masked prompt).
    # We deliberately do NOT assert exact per-position labels — that depends on
    # tokenizer/template boundary behavior the build_sft_example docstring
    # already caveats as not universally stable.
    tok = _FakeTokenizer()
    row = {"messages": [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "general kenobi"},
        {"role": "user", "content": "goodbye"},
        {"role": "assistant", "content": "see you later"},
    ]}
    ex = build_sft_example(row, tok, max_seq_len=4096)
    ids = ex["input_ids"].tolist()
    labels = ex["labels"].tolist()
    assert len(ids) == len(labels), (len(ids), len(labels))
    n_labeled = sum(1 for l in labels if l != -100)
    n_masked = sum(1 for l in labels if l == -100)
    assert n_labeled > 0, "assistant turns produced no labeled tokens"
    assert n_masked > 0, "prompt turns were not masked to -100"
    # Labeled tokens must be a subset of the input ids at the same positions
    # (the mask only hides; it never invents different ids).
    for i, l in enumerate(labels):
        assert l == -100 or l == ids[i], (i, l, ids[i])
    print(f"  OK (len={len(ids)} labeled={n_labeled} masked={n_masked}; "
          f"mask only hides, never rewrites ids)")

    # Empty assistant set -> empty example (fallback edge case).
    empty = build_sft_example({"messages": [{"role": "user", "content": "no reply"}]},
                              tok, 4096)
    assert empty["input_ids"].numel() == 0 or \
        (empty["labels"] == -100).all().item(), "no-assistant row must be fully masked"
    print("  OK (row with no assistant turn yields a fully-masked/empty example)")

    print("\n[selftest] All checks passed (core logic needs no GPU/torch/tokenizer; "
          "torch-only round-trip + masking checks ran since torch is importable).")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=None,
                    help="Input JSONL file ({\"text\":...} or {\"messages\":[...]}). "
                         "Required unless --selftest.")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HF tokenizer dir/repo (e.g. ./checkpoints/base). "
                         "Required unless --selftest.")
    ap.add_argument("--dst", type=Path, default=None,
                    help="Output directory for shard_*.pt files. Required unless "
                         "--selftest. Created if missing.")
    ap.add_argument("--mode", choices=["cpt", "sft"], default="cpt",
                    help="cpt: {\"text\":...} rows, every token is a label. "
                         "sft: {\"messages\":[...]} rows, assistant-turn-only "
                         "labels (-100 on prompt). Default: cpt.")
    ap.add_argument("--shard-size", type=int, default=10000,
                    help="Rows per .pt shard (default 10000). The final shard "
                         "may be shorter.")
    ap.add_argument("--max-seq-len", type=int, default=2048,
                    help="Per-row truncation length in tokens (default 2048). "
                         "Must be >= the trainer's --max-seq-len or rows will "
                         "have been truncated shorter than the trainer expects.")
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py). "
                         "No-op on non-ROCm boxes; tokenization itself is CPU.")
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True",
                    help="PYTORCH_HIP_ALLOC_CONF value (pass 'none' to skip). "
                         "Irrelevant to tokenization (CPU tensors) but kept for "
                         "pipeline consistency.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU/tokenizer required for "
                         "core checks) and exit.")
    args = ap.parse_args()

    if args.selftest:
        self_test()
        return

    if not (args.src and args.tokenizer and args.dst):
        ap.error("--src, --tokenizer, and --dst are required unless --selftest is given.")
    if args.shard_size <= 0:
        ap.error(f"--shard-size must be >= 1 (got {args.shard_size}).")

    run_pretokenize(args.src, args.tokenizer, args.dst, args.mode,
                    args.shard_size, args.max_seq_len,
                    args.gfx_override, args.hip_alloc_conf)


if __name__ == "__main__":
    main()
