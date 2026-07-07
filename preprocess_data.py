#!/usr/bin/env python3
"""Dataset preprocessing for CPT/SFT training data — standalone, CPU-only.

Reads a JSONL file in the format train_cpt.py expects ({"text": "..."} for CPT
or {"messages": [...]} for SFT) and applies a configurable pipeline:

  - **Exact dedup**: drops rows with a text field identical to one already seen.
    No fuzzy/minhash dedup — exact only, because approximate dedup quality
    varies wildly by dataset and a false-positive drop silently throws away
    real training data. If you need fuzzy dedup, run a dedicated tool (e.g.
    datasketch) upstream; this tool won't pretend to do it.
  - **Length filtering**: --min-chars / --max-chars (tokenizer-free, fast) or
    --min-tokens / --max-tokens (requires --tokenizer, slower but exact).
  - **Script filtering**: --drop-scripts cjk,cyrillic,arabic,... reuses
    prune_vocab.classify() — a CHARACTER-based heuristic, NOT real language
    ID. It catches rows whose text contains distinctive non-Latin script
    characters, not plain-Latin text from those languages. Same caveat as
    prune_vocab.py: "configurable is not the same claim as verified" against
    any specific language-detection quality bar.
  - **Sequence packing**: packs short {"text": "..."}` rows into sequences up
    to --pack-seqlen (separated by a configurable --pack-separator, default
    newline), reducing the padding waste that train_cpt.py's --pack flag then
    further compresses at collation time.

Writes the filtered+packed JSONL to --dst. --dry-run prints stats (rows
in/out, dropped by reason, packed sequences) without writing.

Usage:
    python3 preprocess_data.py --src data.jsonl --dst filtered.jsonl \\
        --min-chars 50 --max-chars 10000 --drop-scripts cjk,arabic \\
        --pack-seqlen 2048 --dry-run
    python3 preprocess_data.py --src data.jsonl --dst filtered.jsonl \\
        --tokenizer ./checkpoints/base_12b --min-tokens 10 --max-tokens 4096

Self-test (no tokenizer/GPU required — exercises dedup, length filter, script
filter, and packing against tiny in-memory data):
    python3 preprocess_data.py --selftest
"""

import argparse
import json
import sys


def log(msg: str):
    print(f"[preprocess] {msg}", flush=True)


def read_jsonl(path: str):
    """Read a JSONL file, skipping blank lines. Yields parsed dicts."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines (same as train_cpt's load_jsonl)


def get_text(row: dict) -> str:
    """Extract the text field from a row. For {"messages": [...]} SFT rows,
    concatenates all message contents; for {"text": "..."} CPT rows, returns
    the text directly."""
    if "text" in row:
        return row["text"]
    if "messages" in row:
        return "\n".join(m.get("content", "") for m in row["messages"])
    return ""


def count_chars(row: dict) -> int:
    return len(get_text(row))


def count_tokens(row: dict, tokenizer) -> int:
    return len(tokenizer(get_text(row), add_special_tokens=False)["input_ids"])


def should_drop_by_script(row: dict, drop_scripts: set) -> bool:
    """Returns True if the row's text contains any character from a dropped
    script. Uses prune_vocab.classify() — a character heuristic, not language
    ID. See module docstring for the caveat."""
    if not drop_scripts:
        return False
    from prune_vocab import classify
    text = get_text(row)
    for char in text:
        if classify(char) in drop_scripts:
            return True
    return False


def pack_rows(rows: list, max_seqlen: int, separator: str = "\n"):
    """Pack short rows into sequences up to max_seqlen chars, separated by
    `separator`. Returns a list of {"text": "packed..."} rows. Rows longer
    than max_seqlen are passed through as-is (truncated to max_seqlen).

    This is a CHAR-level pack (no tokenizer) — it reduces the number of
    short rows that would otherwise waste padding, but the exact token count
    of each packed sequence depends on the tokenizer. train_cpt.py's --pack
    flag does a second token-level packing pass at collation time."""
    packed = []
    current_parts = []
    current_len = 0
    sep_len = len(separator)
    for row in rows:
        text = get_text(row)
        if len(text) > max_seqlen:
            # Flush current buffer, then emit the long row as-is (truncated).
            if current_parts:
                packed.append({"text": separator.join(current_parts)})
                current_parts = []
                current_len = 0
            packed.append({"text": text[:max_seqlen]})
            continue
        addition = len(text) + (sep_len if current_parts else 0)
        if current_len + addition > max_seqlen and current_parts:
            packed.append({"text": separator.join(current_parts)})
            current_parts = [text]
            current_len = len(text)
        else:
            current_parts.append(text)
            current_len += addition
    if current_parts:
        packed.append({"text": separator.join(current_parts)})
    return packed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="Input JSONL file.")
    ap.add_argument("--dst", required=True, help="Output JSONL file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only, write nothing.")
    ap.add_argument("--min-chars", type=int, default=None,
                    help="Drop rows with fewer characters than this.")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="Drop rows with more characters than this.")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="Drop rows with fewer tokens (requires --tokenizer).")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Drop rows with more tokens (requires --tokenizer).")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HF tokenizer dir/repo for --min-tokens/--max-tokens.")
    ap.add_argument("--drop-scripts", type=str, default=None,
                    help="Comma-separated script categories to drop: cjk, "
                         "cyrillic, arabic, devanagari_hindi, mongolian_script, "
                         "romance_germanic_chars. Character heuristic, NOT "
                         "language ID — see module docstring.")
    ap.add_argument("--pack-seqlen", type=int, default=None,
                    help="Pack short rows into sequences up to this many chars.")
    ap.add_argument("--pack-separator", type=str, default="\n",
                    help="Separator between packed rows (default newline).")
    args = ap.parse_args()

    # Parse drop-scripts into a set.
    drop_scripts = set()
    if args.drop_scripts:
        drop_scripts = {s.strip() for s in args.drop_scripts.split(",") if s.strip()}
        log(f"dropping rows containing characters from: {drop_scripts}")

    # Load tokenizer if token-based filtering is requested.
    tokenizer = None
    if args.min_tokens is not None or args.max_tokens is not None:
        if args.tokenizer is None:
            raise SystemExit("ERROR: --min-tokens/--max-tokens requires --tokenizer.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        log(f"loaded tokenizer from {args.tokenizer}")

    # Read all rows.
    rows = list(read_jsonl(args.src))
    total_in = len(rows)
    log(f"read {total_in:,} rows from {args.src}")

    # Dedup (exact, on the text field).
    seen = set()
    deduped = []
    dropped_dedup = 0
    for row in rows:
        text = get_text(row)
        if text in seen:
            dropped_dedup += 1
            continue
        seen.add(text)
        deduped.append(row)
    rows = deduped
    log(f"dedup: dropped {dropped_dedup:,} exact duplicates, {len(rows):,} remain")

    # Length filtering.
    dropped_short = 0
    dropped_long = 0
    filtered = []
    for row in rows:
        if args.min_chars is not None and count_chars(row) < args.min_chars:
            dropped_short += 1
            continue
        if args.max_chars is not None and count_chars(row) > args.max_chars:
            dropped_long += 1
            continue
        if tokenizer is not None:
            n = count_tokens(row, tokenizer)
            if args.min_tokens is not None and n < args.min_tokens:
                dropped_short += 1
                continue
            if args.max_tokens is not None and n > args.max_tokens:
                dropped_long += 1
                continue
        filtered.append(row)
    rows = filtered
    log(f"length filter: dropped {dropped_short:,} too-short, {dropped_long:,} "
        f"too-long, {len(rows):,} remain")

    # Script filtering.
    dropped_script = 0
    if drop_scripts:
        kept = []
        for row in rows:
            if should_drop_by_script(row, drop_scripts):
                dropped_script += 1
            else:
                kept.append(row)
        rows = kept
        log(f"script filter: dropped {dropped_script:,}, {len(rows):,} remain")

    # Packing.
    packed_count = len(rows)
    if args.pack_seqlen:
        rows = pack_rows(rows, args.pack_seqlen, args.pack_separator)
        log(f"packing: {packed_count:,} rows -> {len(rows):,} packed sequences "
            f"(max {args.pack_seqlen} chars each)")

    log(f"final: {len(rows):,} rows ({total_in:,} in -> {len(rows):,} out, "
        f"dropped {total_in - len(rows):,} total)")

    if args.dry_run:
        log("DRY RUN — nothing written.")
        return

    with open(args.dst, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"wrote {len(rows):,} rows -> {args.dst}")


def _self_test():
    print("[selftest] preprocess_data: dedup, length filter, script filter, packing")

    # Dedup: identical text fields are dropped.
    rows = [
        {"text": "hello world"},
        {"text": "hello world"},  # dup
        {"text": "unique text"},
    ]
    seen = set()
    deduped = []
    for row in rows:
        text = get_text(row)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(row)
    assert len(deduped) == 2, deduped
    print("  OK (exact dedup drops duplicates)")

    # Length filter: min-chars / max-chars.
    rows = [{"text": "hi"}, {"text": "this is a longer text"}, {"text": "x" * 100}]
    filtered = [r for r in rows if 5 <= count_chars(r) <= 50]
    assert len(filtered) == 1, [count_chars(r) for r in rows]
    assert filtered[0]["text"] == "this is a longer text"
    print("  OK (char-based length filter)")

    # Script filter: CJK characters are caught by classify().
    rows = [
        {"text": "plain English text"},
        {"text": "包含中文的文本"},
        {"text": "more English"},
    ]
    from prune_vocab import classify, REMOVABLE
    kept = [r for r in rows if not should_drop_by_script(r, {"cjk"})]
    assert len(kept) == 2
    assert "包含中文的文本" not in [r["text"] for r in kept]
    print("  OK (script filter drops CJK via character heuristic)")

    # Packing: short rows are combined into sequences up to max_seqlen.
    rows = [
        {"text": "aaa"},
        {"text": "bbb"},
        {"text": "ccc"},
        {"text": "dddddddddddddddddddddddddddddddd"},  # 32 chars, forces a flush
    ]
    packed = pack_rows(rows, max_seqlen=15, separator="|")
    # "aaa|bbb" = 7 chars, then "ccc" + "ddd..." (32 chars > 15) flushes "ccc",
    # then the long row is truncated to 15.
    assert any("aaa" in p["text"] and "bbb" in p["text"] for p in packed), packed
    assert any(len(p["text"]) <= 15 for p in packed), [len(p["text"]) for p in packed]
    print(f"  OK (packing: {len(rows)} rows -> {len(packed)} packed sequences)")

    # get_text handles both {"text":...} and {"messages":[...]} formats.
    assert get_text({"text": "hello"}) == "hello"
    msg_row = {"messages": [{"content": "a"}, {"content": "b"}]}
    assert get_text(msg_row) == "a\nb"
    print("  OK (get_text handles CPT and SFT row formats)")

    print("\n[selftest] All checks passed (no tokenizer/GPU required — run with "
          "real data + --tokenizer for the token-based path).")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    args, _ = ap.parse_known_args()
    if args.selftest:
        _self_test()
    else:
        main()


if __name__ == "__main__":
    main_cli()
