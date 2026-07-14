#!/bin/bash
# Preemptive OOM guard for a training process: poll free system RAM and ALL
# visible GPUs, warn under a soft threshold, and SIGTERM the training process
# under a hard emergency threshold -- so it dies BEFORE the OS/driver hits an
# unrecoverable OOM state, not after.
#
# Supports AMD GPUs via `rocm-smi`. If rocm-smi is not available, the guard
# falls back to system-RAM-only monitoring.
#
# Usage: nohup bash oom_guard.sh <training_pid> [warn_free_mb] [emergency_free_mb] \
#                                   [poll_sec] [vram_warn_mb] [vram_emergency_mb] \
#                                   > oom_guard.log 2>&1 &
# Stop:  kill the guard's own PID (printed at start), or pkill -f oom_guard.sh

set -uo pipefail
TRAIN_PID="${1:?usage: oom_guard.sh <training_pid> [warn_free_mb] [emergency_free_mb] [poll_sec] [vram_warn_mb] [vram_emergency_mb]}"
WARN_FREE_MB="${2:-4000}"
EMERGENCY_FREE_MB="${3:-1500}"
POLL_SEC="${4:-30}"
VRAM_WARN_MB="${5:-2048}"
VRAM_EMERGENCY_MB="${6:-512}"

echo "[oom_guard] watching PID $TRAIN_PID, poll ${POLL_SEC}s"
echo "[oom_guard] system-RAM: warn<${WARN_FREE_MB}MB, emergency<${EMERGENCY_FREE_MB}MB (via /proc/meminfo)"

read_available_mb() {
    local kb
    kb=$(awk '/^MemAvailable:/ {print $2; found=1} END {if (!found) print ""}' /proc/meminfo 2>/dev/null)
    if [ -z "$kb" ]; then
        kb=$(awk '/^MemFree:/ {print $2}' /proc/meminfo 2>/dev/null)
    fi
    if [ -z "$kb" ]; then
        echo ""
        return
    fi
    echo $((kb / 1024))
}

# ---------------------------------------------------------------------------
# GPU backend detection and per-GPU free-VRAM polling.
# ---------------------------------------------------------------------------
GPU_BACKEND="none"
if command -v rocm-smi >/dev/null 2>&1; then
    GPU_BACKEND="rocm"
    echo "[oom_guard] GPU backend: ROCm (rocm-smi)"
else
    echo "[oom_guard] no rocm-smi found -- GPU-VRAM checks skipped (system-RAM only)."
fi

read_all_vram_free_mb() {
    # Prints one line per GPU: "<index> <free_mb>". Empty output means unavailable.
    if [ "$GPU_BACKEND" = "rocm" ]; then
        _read_rocm_vram
    fi
}

_read_rocm_vram() {
    # Try JSON first, then fall back to text. Aggregate free memory per GPU.
    #
    # BUG FIX: the previous version piped $json_out / $raw into `python3 -`
    # while ALSO using a `<<'PY' ... PY` heredoc on the same command. A
    # heredoc redirects stdin too, and it wins over the pipe -- so the
    # embedded script's sys.stdin.read() always saw an EMPTY string (verified
    # directly: `echo "$x" | python3 - <<'PY' ... print(repr(sys.stdin.read()))
    # PY` prints `''`), json.load()/manual parsing then failed every single
    # time, and the JSON branch's unconditional `return` meant the (correctly
    # written) text fallback below was never even reached. Net effect: VRAM
    # monitoring was silently a permanent no-op on every real ROCm box (where
    # `rocm-smi --json` always succeeds) -- the emergency SIGTERM this whole
    # script exists for would never fire from VRAM pressure. Fixed by passing
    # the captured output as an argv argument instead of over stdin, so it
    # doesn't collide with the heredoc. Also made the JSON branch fall through
    # to the text fallback when it produces no rows (e.g. truly malformed
    # rocm-smi output), instead of returning unconditionally.
    local json_out
    json_out=$(timeout 10 rocm-smi --showmeminfo vram --json 2>/dev/null)
    if [ -n "$json_out" ]; then
        local json_result
        json_result=$(python3 -c '
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)
for card, info in data.items():
    idx = int(card.replace("card", "")) if card.startswith("card") else 0
    free_b = info.get("VRAM Free Memory (B)", 0)
    if isinstance(free_b, (int, float)) and free_b > 0:
        print(idx, int(free_b / 1024 / 1024))
' "$json_out" 2>/dev/null)
        if [ -n "$json_result" ]; then
            echo "$json_result"
            return
        fi
        # Fall through to the text fallback if JSON parsed to zero rows.
    fi

    # Text fallback: each GPU block has "VRAM Free Memory (B): <bytes>".
    local raw
    raw=$(timeout 10 rocm-smi --showmeminfo vram 2>/dev/null)
    if [ -z "$raw" ]; then
        return
    fi
    python3 -c '
import re, sys
text = sys.argv[1]
# Split by GPU headers like "GPU[0]"
gpus = re.split(r"GPU\[(\d+)\]", text)
# First element is preamble; then pairs of (index, block).
for i in range(1, len(gpus), 2):
    idx = gpus[i]
    block = gpus[i + 1] if i + 1 < len(gpus) else ""
    m = re.search(r"VRAM Free Memory \(B\):\s*(\d+)", block)
    if m:
        print(idx, int(m.group(1)) // 1024 // 1024)
' "$raw"
}

lowest_vram_free_mb() {
    local min_free=""
    while read -r idx free_mb; do
        if [ -z "$min_free" ] || [ "$free_mb" -lt "$min_free" ]; then
            min_free="$free_mb"
        fi
    done < <(read_all_vram_free_mb)
    echo "$min_free"
}

# ---------------------------------------------------------------------------
# Main poll loop.
# ---------------------------------------------------------------------------
while true; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[oom_guard] $(date '+%H:%M:%S') training PID $TRAIN_PID no longer exists -- exiting guard."
        exit 0
    fi

    free_mb=$(read_available_mb)
    if [ -z "$free_mb" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') could not read /proc/meminfo -- skipping this poll"
        sleep "$POLL_SEC"
        continue
    fi

    if [ "$free_mb" -lt "$EMERGENCY_FREE_MB" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') EMERGENCY: only ${free_mb}MB system RAM available -- sending SIGTERM to $TRAIN_PID."
        kill -TERM "$TRAIN_PID" 2>/dev/null
    elif [ "$free_mb" -lt "$WARN_FREE_MB" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') WARNING: ${free_mb}MB system RAM available -- getting tight."
    fi

    if [ "$GPU_BACKEND" != "none" ]; then
        vram_free_mb=$(lowest_vram_free_mb)
        if [ -n "$vram_free_mb" ]; then
            if [ "$vram_free_mb" -lt "$VRAM_EMERGENCY_MB" ]; then
                echo "[oom_guard] $(date '+%H:%M:%S') EMERGENCY: only ${vram_free_mb}MB VRAM free across GPUs -- sending SIGTERM to $TRAIN_PID."
                kill -TERM "$TRAIN_PID" 2>/dev/null
            elif [ "$vram_free_mb" -lt "$VRAM_WARN_MB" ]; then
                echo "[oom_guard] $(date '+%H:%M:%S') WARNING: ${vram_free_mb}MB VRAM free across GPUs -- getting tight."
            fi
        fi
    fi

    sleep "$POLL_SEC"
done
