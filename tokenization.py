"""Shared tokenization builders for SFT and CPT training data.

Extracted from train_cpt.py so pretokenize.py imports the SAME functions
instead of maintaining a verbatim copy that can silently diverge.

Usage:
    from tokenization import build_sft_example, build_cpt_example
"""


def build_sft_example(row: dict, tokenizer, max_seq_len: int):
    """Chat-template tokenize with prompt masking: only assistant-turn tokens get a
    real label, everything else (system/user/special tokens) is -100 (ignored by
    cross-entropy).

    Implementation note: this tokenizes incrementally, calling
    apply_chat_template(messages[:i+1]) per turn and diffing against the previous
    turn's rendered text to isolate the new span. This is O(n_turns^2) in template
    applications per example, and it assumes the template is strictly appenditive —
    i.e. apply_chat_template(messages[:i+1]) is a verbatim text prefix of
    apply_chat_template(messages[:i+2]). This holds for Gemma-4's template (what
    this pipeline targets) but NOT universally; templates that re-render based on
    the full message list, or emit a trailing EOS/generation marker only at the
    end, break the prefix assumption and would silently mis-tokenize/mis-label. We
    detect that break (the prefix check below) and fall back to a single full
    tokenization with the whole prompt masked — coarser (loses per-turn assistant
    labeling, labels only the last assistant turn) and approximate (assumes the
    prompt tokenization is a token-level prefix of the full text, which isn't
    guaranteed for non-appenditive templates), rather than silently wrong.
    """
    import torch

    messages = row["messages"]
    input_ids: list[int] = []
    labels: list[int] = []

    # Tokenize turn-by-turn so we know exactly which spans are assistant output.
    running_text = ""
    prefix_assumption_holds = True
    for i, msg in enumerate(messages):
        prefix_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        # Detect a non-appenditive template: if the new full text doesn't start
        # with the previous full text, the incremental-diff approach is invalid.
        if not prefix_text.startswith(running_text):
            prefix_assumption_holds = False
            break
        new_text = prefix_text[len(running_text):]
        running_text = prefix_text
        ids = tokenizer(new_text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        if msg["role"] == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))

    if not prefix_assumption_holds:
        # Fallback: tokenize the full conversation once, mask everything before
        # the last assistant turn, label only the last assistant turn's tokens.
        # This is APPROXIMATE for non-appenditive templates — it assumes the
        # prompt-text tokenization is a token-level prefix of the full-text
        # tokenization, which isn't guaranteed for templates that re-render.
        # It's safer than the broken incremental diff (which would silently
        # mis-tokenize), but it only labels the LAST assistant turn, not all
        # of them. Gemma-4's template is appenditive and takes the primary path
        # above, so this fallback rarely runs for the targeted model family.
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Build the prompt = everything up to the last assistant turn, to mask it.
        last_assistant_idx = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"),
            default=-1,
        )
        if last_assistant_idx == -1:
            return {"input_ids": torch.tensor([], dtype=torch.long),
                    "labels": torch.tensor([], dtype=torch.long)}
        prompt_text = tokenizer.apply_chat_template(
            messages[:last_assistant_idx] if last_assistant_idx >= 0 else [],
            tokenize=False, add_generation_prompt=True,
        ) if last_assistant_idx >= 0 else ""
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        p_len = len(prompt_ids)
        input_ids = full_ids
        labels = [-100] * min(p_len, len(full_ids)) + full_ids[min(p_len, len(full_ids)):]
        labels = labels[:len(full_ids)]

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_cpt_example(row: dict, tokenizer, max_seq_len: int):
    """Raw-text CPT: every token is a label (no masking). Expects packed
    {"text": "..."} rows."""
    import torch

    text = row.get("text", "")
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                     max_length=max_seq_len)["input_ids"]
    t = torch.tensor(ids, dtype=torch.long)
    return {"input_ids": t, "labels": t.clone()}
