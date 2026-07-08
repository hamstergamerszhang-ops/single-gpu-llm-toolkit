#!/bin/bash
# Auto-resuming supervisor for train_cpt.py.
#
# Adapted from a proven MLX-side supervisor pattern used earlier in this
# project for an Apple-Silicon training script (loss-tagged checkpoint
# history + rollback-on-regression + retry-with-backoff + a stop-file). That
# original script drove a DIFFERENT training script with a DIFFERENT resume
# contract: it passed explicit `--resume <checkpoint_path> --start-iter N`
# flags on every relaunch, because that script's checkpoint filenames used a
# LOCAL step counter that reset to 1 on every invocation, so the supervisor
# had to track global position itself and hand back an explicit resume
# target every time.
#
# train_cpt.py (this repo) does NOT work that way, and this script is written
# against train_cpt.py's REAL, actual flags — checked directly against its
# argparse section, not assumed from the older script:
#   - There is no --resume or --start-iter flag on train_cpt.py at all.
#   - train_cpt.py self-resumes: on startup it checks whether
#     `<save_dir>/training_state.pt` already exists, and if so, loads model +
#     optimizer + step count from there automatically — no flag needed. Just
#     re-running the exact same command after a crash resumes correctly.
#   - Checkpoints are written to a SINGLE path (--save), overwritten in place
#     (atomically) every --checkpoint-every steps — there's no per-step
#     numbered checkpoint file the way the MLX original had.
#
# Because of that, this supervisor is simpler than the MLX-side one in one
# respect (no need to compute/pass a resume target — just relaunch the same
# command) but adds its own value on top of train_cpt.py's built-in resume:
#   - Loss-tagged checkpoint HISTORY, so a bad patch of training (e.g. a data
#     issue that spikes the loss) can be rolled back to a known-good point,
#     not just "whatever train_cpt.py currently has on disk" (which is a
#     single slot that a bad patch would silently overwrite).
#   - Retry-with-backoff + a same-position stall detector, so a genuinely
#     stuck/looping crash doesn't retry forever silently.
#   - A stop-file so you can request a clean stop without killing the process
#     mid-write.
#
# Usage:   ./catch_and_resume.sh
# Stop early: touch .stop_autoresume (checked between attempts)

set -uo pipefail
cd "$(dirname "$0")"

# Config: sourced from config.env if present (see config.env.example), so you
# edit config.env rather than this script. Falls back to built-in defaults
# matching config.env.example if config.env doesn't exist.
# Define ALL defaults first, then source config.env on top to override. This
# way a partial config.env (missing some vars) doesn't crash under set -u.
MODEL=./checkpoints/base_expanded_15b
DATA=./data/data_cpt_1
CPT_CACHE=
SAVE=./checkpoints/model_cpt_1
TOTAL_ITERS=10000
CHECKPOINT_EVERY=500
BATCH=4
LR=5e-7
WARMUP_STEPS=50
MAX_SEQ_LEN=2048
STOP_FILE=.stop_autoresume
LOG_PREFIX=./logs/cpt_1_autoresume
HISTORY_KEEP=4
LOSS_REGRESSION_FACTOR=1.5
MAX_SAME_POSITION_RETRIES=8
RETRY_SLEEP_SECS=10

CONFIG_FILE="${1:-config.env}"
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck source=config.env
    source "$CONFIG_FILE"
    echo "[autoresume] loaded config from $CONFIG_FILE (defaults overridden where set)"
else
    echo "[autoresume] WARNING: $CONFIG_FILE not found -- using built-in defaults. "
    echo "[autoresume] Copy config.env.example to config.env and edit it for your run."
fi

HISTORY_DIR="${SAVE}_history"

mkdir -p "$HISTORY_DIR" "$(dirname "$LOG_PREFIX")" "$(dirname "$SAVE")"

attempt=0
same_position_retries=0
last_seen_step=-1

# "Current step" is read straight from train_cpt.py's own checkpoint --
# there is no separate state file to keep in sync, unlike a supervisor for a
# script whose checkpoint filenames reset their own counter every run.
read_current_step() {
    python3 - "$SAVE" <<'PYEOF'
import sys
import torch
from pathlib import Path

save_dir = Path(sys.argv[1])
state_path = save_dir / "training_state.pt"
if not state_path.exists():
    print(0)
else:
    try:
        # weights_only=False matches train_cpt.py's resume path: training_state.pt
        # holds an optimizer state_dict with non-allowlisted pickle objects, which
        # torch >= 2.6's default weights_only=True would reject. PyTorch 2.6+
        # defaults to weights_only=True, breaking resume without this.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        print(state.get("step", 0))
    except Exception:
        # Corrupt/partial checkpoint: print 0 so the supervisor treats it as
        # "no progress" rather than crashing with a traceback that leaves
        # current_step empty and causes an "integer expression expected" cascade.
        print(0)
PYEOF
}

# Read the valid_loss field train_cpt.py writes into training_state.pt's
# extra_state (when a held-out valid.jsonl is present). Falls back to empty
# string if absent (e.g. --no-eval, or no valid.jsonl). Used to prefer held-out
# loss over training loss for rollback decisions — train loss hides overfitting.
read_valid_loss() {
    python3 - "$1" <<'PYEOF'
import sys
import torch
from pathlib import Path

save_dir = Path(sys.argv[1])
state_path = save_dir / "training_state.pt"
if not state_path.exists():
    print("")
else:
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        vl = state.get("valid_loss", None)
        print(vl if vl is not None else "")
    except Exception:
        print("")
PYEOF
}

while true; do
    if [ -f "$STOP_FILE" ]; then
        echo "[autoresume] Stop file found, exiting. Resume later by removing $STOP_FILE."
        exit 0
    fi

    current_step=$(read_current_step)
    if [ "$current_step" -ge "$TOTAL_ITERS" ]; then
        echo "[autoresume] Reached target $TOTAL_ITERS iterations (checkpoint reports step $current_step)."
        exit 0
    fi

    attempt=$((attempt + 1))
    log_file="${LOG_PREFIX}_attempt${attempt}.log"
    echo "[autoresume] Attempt $attempt: launching train_cpt.py (checkpoint currently at "\
"step $current_step / $TOTAL_ITERS) -> $log_file"

    data_args=()
    if [ -n "$CPT_CACHE" ]; then
        data_args=(--cpt-cache "$CPT_CACHE")
    else
        data_args=(--data "$DATA")
    fi

    # NOTE: no --resume / --start-iter here, unlike the MLX-side supervisor this
    # pattern is adapted from -- train_cpt.py finds $SAVE/training_state.pt itself
    # and resumes from it automatically. Re-running the identical command is the
    # entire resume mechanism.
    python3 train_cpt.py \
        --model "$MODEL" \
        --save "$SAVE" \
        "${data_args[@]}" \
        --cpt \
        --iters "$TOTAL_ITERS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --warmup-steps "${WARMUP_STEPS:-50}" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --async-checkpoint \
        > "$log_file" 2>&1
    exit_code=$?

    new_step=$(read_current_step)

    if [ "$exit_code" -eq 0 ] && [ "$new_step" -ge "$TOTAL_ITERS" ]; then
        echo "[autoresume] Run completed cleanly (exit 0), checkpoint at step $new_step. Training complete."
        exit 0
    fi

    echo "[autoresume] Attempt $attempt exited with code $exit_code (step $current_step -> $new_step). Last 5 log lines:"
    tail -5 "$log_file"

    if [ "$new_step" -le "$last_seen_step" ] && [ "$last_seen_step" -ge 0 ]; then
        same_position_retries=$((same_position_retries + 1))
        if [ "$same_position_retries" -ge "$MAX_SAME_POSITION_RETRIES" ]; then
            echo "[autoresume] No new checkpoint progress after $same_position_retries retries "\
"at step $new_step. Stopping -- this looks like a real recurring problem, not transient pressure."
            exit 1
        fi
        echo "[autoresume] Crashed without advancing past the last checkpoint (still at step "\
"$new_step). Retrying -- attempt $same_position_retries/$MAX_SAME_POSITION_RETRIES."
    else
        same_position_retries=0

        # Tag this checkpoint with its loss and copy it into the loss-tagged
        # history (a COPY, not a move -- train_cpt.py's own $SAVE path stays
        # where it is and keeps getting overwritten by the next checkpoint) so a
        # later bad-patch checkpoint can never destroy this one.
        #
        # Prefer valid_loss (held-out, written into training_state.pt by
        # train_cpt.py's eval loop) over training-loss grep: train loss hides
        # overfitting, which is exactly the failure rollback exists to catch.
        last_loss=$(read_valid_loss "$SAVE")
        loss_kind="valid"
        if [ -z "$last_loss" ]; then
            # No held-out eval (--no-eval, no valid.jsonl, or eval not yet run at
            # this checkpoint) — fall back to the last logged training loss.
            last_loss=$(grep -oE "loss=[0-9.]+" "$log_file" | tail -1 | grep -oE "[0-9.]+$")
            loss_kind="train"
        fi
        if [ -n "$last_loss" ] && [ -d "$SAVE" ]; then
            hist_entry="$HISTORY_DIR/step${new_step}"
            rm -rf "$hist_entry"
            if ! cp -r "$SAVE" "$hist_entry"; then
                echo "[autoresume] ERROR: history copy failed"
                rm -rf "$hist_entry"
                continue
            fi
            # NOTE: .train_loss may hold valid_loss OR train_loss depending on
            # which was available (see $loss_kind above). The rollback
            # comparison below compares these numerically — if eval cadence
            # differs across checkpoints, this can mix scales. For consistent
            # rollback, run eval at every checkpoint (--eval-every = --checkpoint-every).
            echo "$last_loss" > "$hist_entry/.train_loss"
            echo "$loss_kind" > "$hist_entry/.loss_kind"
            echo "[autoresume] History: saved step $new_step checkpoint with $loss_kind loss $last_loss"
        else
            echo "[autoresume] WARNING: could not determine loss for step $new_step -- "\
"history entry skipped (rollback won't see this checkpoint)."
        fi

        # Prune history beyond HISTORY_KEEP, oldest step first.
        ls -d "$HISTORY_DIR"/step* 2>/dev/null | sed -E 's#.*/step([0-9]+)$#\1 &#' \
            | sort -rn | awk '{print $2}' | tail -n +$((HISTORY_KEEP + 1)) \
            | while read -r old_dir; do
                echo "[autoresume] Pruning history entry: $old_dir"
                rm -rf "$old_dir"
            done
    fi

    # Loss-regression rollback: if the checkpoint train_cpt.py is about to resume
    # from has a train loss much worse than the best one still in history, swap
    # in the better history entry BEFORE the next relaunch instead of letting
    # train_cpt.py keep building on a regression.
    best_dir=""
    best_loss=""
    for loss_file in "$HISTORY_DIR"/step*/.train_loss; do
        [ -e "$loss_file" ] || continue
        loss_val=$(cat "$loss_file")
        if [ -z "$best_loss" ] || awk -v a="$loss_val" -v b="$best_loss" 'BEGIN{exit !(a<b)}'; then
            best_loss="$loss_val"
            best_dir=$(dirname "$loss_file")
        fi
    done
    current_loss_file="$HISTORY_DIR/step${new_step}/.train_loss"
    if [ -n "$best_dir" ] && [ -e "$current_loss_file" ]; then
        current_loss=$(cat "$current_loss_file")
        # Read the loss kind from the sidecar file (may be "valid" or "train")
        # rather than relying on the loop-scoped $loss_kind which may be stale.
        current_loss_kind="train"
        if [ -e "$HISTORY_DIR/step${new_step}/.loss_kind" ]; then
            current_loss_kind=$(cat "$HISTORY_DIR/step${new_step}/.loss_kind")
        fi
        is_regression=$(awk -v cur="$current_loss" -v best="$best_loss" -v factor="$LOSS_REGRESSION_FACTOR" \
            'BEGIN{print (cur > best*factor) ? 1 : 0}')
        if [ "$is_regression" -eq 1 ] && [ "$best_dir" != "$HISTORY_DIR/step${new_step}" ]; then
            echo "[autoresume] Current checkpoint (step $new_step) has $current_loss_kind loss $current_loss, "\
"more than ${LOSS_REGRESSION_FACTOR}x the best kept loss $best_loss ($best_dir) -- "\
"rolling back to the better checkpoint instead of compounding a bad patch."
            rm -rf "$SAVE"
            if ! cp -r "$best_dir" "$SAVE"; then
                # Hard-stop, don't limp on: $SAVE is now empty/missing (the rm -rf
                # above already ran), and letting the loop continue to the next
                # relaunch would mean train_cpt.py finds no training_state.pt and
                # silently restarts from --model at step 0 -- discarding the ENTIRE
                # loss-tagged history this rollback exists to protect, not just the
                # one bad checkpoint it was trying to roll back from. That's a much
                # worse outcome than stopping and making the disk-space (or
                # permissions) problem visible immediately.
                echo "[autoresume] FATAL: rollback copy failed — $SAVE is now empty/missing " \
"after rm -rf (best_dir=$best_dir). Refusing to continue: the next relaunch would " \
"silently restart from --model (step 0), discarding all training progress instead of " \
"just the regressed checkpoint. Check disk space / permissions on $(dirname "$SAVE") " \
"and $HISTORY_DIR, then restore a checkpoint into $SAVE manually (e.g. 'cp -r " \
"$best_dir $SAVE') before re-running this script." >&2
                exit 1
            fi
        fi
    fi

    last_seen_step="$new_step"
    echo "[autoresume] Retrying in ${RETRY_SLEEP_SECS}s..."
    sleep "$RETRY_SLEEP_SECS"
done
