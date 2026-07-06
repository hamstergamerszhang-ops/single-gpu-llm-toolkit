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
#   - GPU-side (VRAM) check: NOT implemented here. `rocm-smi --showmeminfo
#     vram` is the natural ROCm equivalent of what you'd want to poll for
#     VRAM specifically, but this port was written without access to actual
#     ROCm hardware to verify rocm-smi's exact output format, parsing
#     behavior, or how reliably it reflects true "about to OOM" pressure
#     under concurrent load. Guessing at that parsing here would be exactly
#     the kind of unverified claim this repo avoids -- see the commented
#     extension point below instead of a fabricated implementation.
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
# Usage: nohup bash oom_guard.sh <training_pid> > oom_guard.log 2>&1 &
# Stop:  kill the guard's own PID (printed at start), or pkill -f oom_guard.sh

set -u
TRAIN_PID="${1:?usage: oom_guard.sh <training_pid> [warn_free_mb] [emergency_free_mb] [poll_sec]}"
WARN_FREE_MB="${2:-4000}"
EMERGENCY_FREE_MB="${3:-1500}"
POLL_SEC="${4:-30}"

echo "[oom_guard] watching PID $TRAIN_PID, warn<${WARN_FREE_MB}MB, emergency<${EMERGENCY_FREE_MB}MB, poll ${POLL_SEC}s"
echo "[oom_guard] system-RAM check only (via /proc/meminfo) -- see this script's header "
echo "[oom_guard] comment for why a GPU-VRAM-side rocm-smi check is NOT included: not "
echo "[oom_guard] verified against real ROCm hardware, so not guessed at here."

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
        echo "[oom_guard] $(date '+%H:%M:%S') EMERGENCY: only ${free_mb}MB available -- sending SIGTERM to $TRAIN_PID."
        kill -TERM "$TRAIN_PID" 2>/dev/null
    elif [ "$free_mb" -lt "$WARN_FREE_MB" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') WARNING: ${free_mb}MB available -- getting tight."
    fi

    # --- GPU-side (VRAM) extension point, NOT implemented (see header) ---
    # If you're on an AMD ROCm box and want to extend this to also watch
    # VRAM headroom directly (not just system RAM), `rocm-smi --showmeminfo
    # vram` is the tool to start from -- but parse and threshold its actual
    # output yourself against your own hardware first; this repo isn't
    # claiming a verified format for it.
    #
    #   vram_line=$(rocm-smi --showmeminfo vram 2>/dev/null)
    #   # ... parse $vram_line for a free/used MB figure, threshold same as above ...

    sleep "$POLL_SEC"
done
