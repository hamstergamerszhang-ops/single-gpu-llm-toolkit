#!/bin/bash
# Preemptive OOM guard for a single-GPU training process: poll free memory,
# warn under a soft threshold, SIGTERM the training process under a hard
# emergency threshold -- so it dies BEFORE the OS/driver hits an
# unrecoverable OOM state, not after.
#
# Ported from a Mac/Metal version (mem_watcher.sh) built after a real
# kernel panic: concurrent GPU-memory pressure from two processes on the
# same box (a training run and a separate inference server that wasn't
# supposed to wake mid-run) corrupted the GPU driver's memory refcounting
# badly enough to panic the kernel. This script's job is to make sure that
# never gets the chance to happen again -- not by predicting the exact
# failure, but by killing the trainer early whenever memory gets
# dangerously tight, on the assumption that a hard kill now is always
# cheaper than an OS-level crash later.
#
# What changed porting Mac -> generic single-GPU (AMD ROCm included):
#   - System-RAM check: the Mac original parsed `top -l 1`'s "PhysMem" line
#     (a macOS-only command/format). That's swapped below for a read of
#     /proc/meminfo's MemAvailable field, which exists on any Linux box
#     (the realistic target for an AMD ROCm training server) and is a
#     better number than MemFree alone -- MemAvailable already accounts for
#     reclaimable cache/buffers, so it doesn't cry wolf over memory the
#     kernel would happily hand back under real pressure.
#   - GPU-side (VRAM) check: IMPLEMENTED via `rocm-smi --showmeminfo vram`.
#     The earlier version of this script left VRAM polling as a commented-out
#     extension point because it hadn't been verified against real ROCm
#     hardware. It now parses rocm-smi's JSON output (preferred, structured)
#     with a text-output fallback, converts bytes->MB, and applies the same
#     warn/emergency-threshold pattern as the system-RAM check. If rocm-smi
#     isn't on PATH or parsing fails, it logs once and skips VRAM checks
#     (keeps the system-RAM check working on non-ROCm boxes) rather than
#     crashing the guard.
#
# What's unchanged from the original (still true, still intentional): the
# wrapped training process is assumed to have NO SIGTERM handler wired to
# anything smarter than "exit" (train_cpt.py in this repo actually DOES
# install a SIGTERM handler that checkpoints before exiting cleanly -- see
# its _on_sigterm -- so pairing this guard with train_cpt.py gets you a
# real clean-save-then-exit, not just a hard kill). For a process with no
# such handler, this is a hard, immediate kill, not a clean save. That's
# accepted deliberately: the goal is to stop BEFORE memory pressure drives
# the OS/driver into an unrecoverable state, not to guarantee a graceful
# shutdown after the fact. Worst-case loss is bounded by however often you
# checkpoint (e.g. train_cpt.py's --checkpoint-every), which is cheap
# insurance against a full crash.
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
echo "[oom_guard] GPU-VRAM:   warn<${VRAM_WARN_MB}MB, emergency<${VRAM_EMERGENCY_MB}MB (via rocm-smi, if available)"

read_available_mb() {
    # /proc/meminfo's MemAvailable is in kB; convert to whole MB. Falls back to
    # MemFree if MemAvailable isn't present (older kernels), which is more
    # conservative (MemFree ignores reclaimable cache, so it under-reports
    # truly available memory -- safer direction to be wrong in for an OOM guard).
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

# Check rocm-smi availability ONCE in the parent shell (not in a subshell).
# read_vram_free_mb is called via $(...) which runs in a subshell, so a flag
# set inside it is lost. Instead, we skip VRAM polling entirely if rocm-smi
# is absent, and warn once here.
ROCM_SMI_AVAILABLE=0
if command -v rocm-smi >/dev/null 2>&1; then
    ROCM_SMI_AVAILABLE=1
else
    echo "[oom_guard] rocm-smi not on PATH -- GPU-VRAM checks skipped (system-RAM only)." >&2
fi

read_vram_free_mb() {
    # Returns free VRAM in MB via rocm-smi, or empty string if unavailable.
    # Tries JSON output first (structured, robust), falls back to text parsing.
    # rocm-smi --showmeminfo vram --json emits e.g.:
    #   { "card0": { "VRAM Total Memory (B)": 17163091968, "VRAM Total Used Memory (B)": 1234567, "VRAM Free Memory (B)": 17141857401 } }
    # The text variant prints lines like "VRAM Free Memory (B): 17141857401".
    if [ "$ROCM_SMI_AVAILABLE" -eq 0 ]; then
        echo ""
        return
    fi

    # Try JSON first.
    local json_bytes
    json_bytes=$(timeout 10 rocm-smi --showmeminfo vram --json 2>/dev/null \
        | grep -oE '"VRAM Free Memory \(B\)": *[0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$json_bytes" ]; then
        echo $((json_bytes / 1024 / 1024))
        return
    fi

    # Fallback: text output. "VRAM Free Memory (B): <number>" — the byte value
    # is the LAST number on the line (anchored to end), NOT the first, because
    # lines often look like "GPU[0] : VRAM Free Memory (B): 17141857401" and
    # head -1 / first-digit would grab the "0" from "GPU[0]" -> 0 MB -> false
    # emergency SIGTERM.
    local text_bytes
    text_bytes=$(timeout 10 rocm-smi --showmeminfo vram 2>/dev/null \
        | grep -E "VRAM Free Memory" | grep -oE '[0-9]+$' | head -1)
    if [ -n "$text_bytes" ]; then
        echo $((text_bytes / 1024 / 1024))
        return
    fi

    # rocm-smi exists but parsing failed — just return empty (skip this poll).
    echo ""
}

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

    # GPU-side (VRAM) check via rocm-smi. Skips silently (after one warning)
    # if rocm-smi isn't available -- keeps the system-RAM check working on
    # non-ROCm boxes. This is the failure mode that actually matters for GPU
    # training: VRAM OOM kills the process or corrupts the driver, and the
    # system-RAM check alone can't see it coming.
    vram_free_mb=$(read_vram_free_mb)
    if [ -n "$vram_free_mb" ]; then
        if [ "$vram_free_mb" -lt "$VRAM_EMERGENCY_MB" ]; then
            echo "[oom_guard] $(date '+%H:%M:%S') EMERGENCY: only ${vram_free_mb}MB VRAM free -- sending SIGTERM to $TRAIN_PID."
            kill -TERM "$TRAIN_PID" 2>/dev/null
        elif [ "$vram_free_mb" -lt "$VRAM_WARN_MB" ]; then
            echo "[oom_guard] $(date '+%H:%M:%S') WARNING: ${vram_free_mb}MB VRAM free -- getting tight."
        fi
    fi

    sleep "$POLL_SEC"
done
