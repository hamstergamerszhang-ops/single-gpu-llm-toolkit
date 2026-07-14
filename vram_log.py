#!/usr/bin/env python3
"""VRAM time-series logger for training runs.

A background process that samples GPU memory every --interval seconds while a
training run is in progress and writes a CSV time-series. This is the
read-only sibling of oom_guard.sh: it LOGS the actual VRAM curve (when the
optimizer state finishes allocating lazily, when eval spikes memory, when a
checkpoint save reserves a big buffer) instead of killing the process.

Two sources are sampled each tick:

  1. torch.cuda.memory_allocated() / torch.cuda.memory_reserved() -- via a
     short-lived Python SUBPROCESS that imports torch and prints both numbers
     in bytes. The logger process itself NEVER imports torch, so it does not
     create a CUDA context that could fragment VRAM or perturb the training
     run. HONESTY NOTE: torch's caching allocator is per-process, so a fresh
     sampler subprocess that holds no model weights will read ~0 for both of
     these columns. They are included as a torch-runtime reachability probe
     (and because the design deliberately isolates the torch import in a
     throwaway subprocess), NOT as the training process's allocator stats --
     this logger cannot read another process's torch allocator from outside.
     To capture the training process's OWN allocator curve, instrument the
     training script and log torch.cuda.memory_allocated() there; this tool's
     torch columns will be ~0 when run as a separate process.

  2. rocm-smi --showmeminfo vram --json -- system-level free VRAM. This DOES
     reflect the training process's real consumption at the device level (it
     is the column to plot for the actual VRAM curve). Falls back to an empty
     value if rocm-smi is absent (CPU-only hosts), so the CSV still records a
     row with the torch probe columns.

CSV columns:
    timestamp, pid, cuda_allocated_mb, cuda_reserved_mb, rocm_vram_free_mb

The logger exits on its own when the watched PID disappears (like oom_guard.sh),
so it is safe to launch under nohup and forget about it.

Usage:
    nohup python3 vram_log.py --pid <training_pid> --output vram.csv --interval 5 &

Self-test (no GPU, no rocm-smi required -- exercises the CSV writer +
rocm-smi JSON parser with mock data, and the graceful-fallback path):
    python3 vram_log.py --selftest
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time


def log(msg: str):
    print(f"[vram_log] {msg}", flush=True)


CSV_HEADER = [
    "timestamp",
    "pid",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "rocm_vram_free_mb",
]

# Inline script the sampler subprocess runs. Prints two lines (allocated bytes,
# reserved bytes) or "UNAVAILABLE" for either if torch/CUDA can't be reached.
# Kept as a single -c string so the subprocess needs no local files and the
# logger process never has to import torch itself.
_TORCH_PROBE = (
    "import sys\n"
    "try:\n"
    "    import torch\n"
    "    if not torch.cuda.is_available():\n"
    "        raise RuntimeError('cuda not available')\n"
    "    print(torch.cuda.memory_allocated())\n"
    "    print(torch.cuda.memory_reserved())\n"
    "except Exception:\n"
    "    print('UNAVAILABLE')\n"
    "    print('UNAVAILABLE')\n"
)


def parse_rocm_smi_json(text):
    """Parse the JSON output of `rocm-smi --showmeminfo vram --json`.

    Returns the MINIMUM free VRAM across all visible cards (bytes), or None if
    no card's free value could be parsed. The minimum (rather than the total)
    is reported so that on multi-GPU boxes the curve tracks the tightest card
    -- the same convention oom_guard.sh uses for its emergency threshold.

    Robust to rocm-smi field-name drift across versions: any key containing
    both 'VRAM' and 'Free' (case-sensitive, matching every rocm-smi variant
    seen in the wild: 'VRAM Free Memory (B)') is accepted.
    """
    if not text or not text.strip():
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    free_values = []
    for _card, info in data.items():
        if not isinstance(info, dict):
            continue
        for key, val in info.items():
            if "VRAM" in key and "Free" in key:
                try:
                    b = int(val)
                except (TypeError, ValueError):
                    continue
                if b > 0:
                    free_values.append(b)
    if not free_values:
        return None
    return min(free_values)


def sample_rocm_vram_free_bytes(timeout=10.0):
    """Run `rocm-smi --showmeminfo vram --json` and return the minimum free
    VRAM across cards (bytes). Returns None if rocm-smi is absent, exits
    non-zero, returns malformed output, or times out. Never raises."""
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        # No rocm-smi on this host (CPU-only). Expected on dev boxes;
        # not an error -- the column is just left empty.
        return None
    except subprocess.TimeoutExpired:
        log("rocm-smi timed out -- skipping this tick's system VRAM sample")
        return None
    if proc.returncode != 0:
        return None
    return parse_rocm_smi_json(proc.stdout)


def sample_torch_cuda(python_exe=None, timeout=15.0):
    """Spawn a subprocess that imports torch and reads the CUDA caching
    allocator. Returns (allocated_bytes, reserved_bytes) as ints, or
    (None, None) if torch/CUDA is unavailable, the probe times out, or the
    interpreter can't be launched. Never raises.

    See the module docstring for the honesty caveat: these are the SAMPLER
    subprocess's own allocator stats (~0), not the training process's.
    """
    exe = python_exe or sys.executable
    try:
        proc = subprocess.run(
            [exe, "-c", _TORCH_PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        log(f"torch probe interpreter not found: {exe!r}")
        return None, None
    except subprocess.TimeoutExpired:
        log("torch probe subprocess timed out -- skipping this tick's CUDA sample")
        return None, None

    lines = (proc.stdout or "").strip().splitlines()
    if len(lines) < 2:
        return None, None

    def _to_int(s):
        s = s.strip()
        if s == "UNAVAILABLE":
            return None
        try:
            return int(s)
        except (TypeError, ValueError):
            return None

    return _to_int(lines[0]), _to_int(lines[1])


def _mb(b):
    """Convert bytes to a whole-MB string, or '' if None (so the CSV column
    is simply empty rather than 'None', which pandas reads as NaN cleanly)."""
    if b is None:
        return ""
    return str(int(b) // (1024 * 1024))


def pid_alive(pid):
    """True if `pid` exists and is signalable from this process."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but not ours to signal. Treat as not-watchable so the logger
        # doesn't hang forever on a PID it can't actually observe.
        return False
    except OSError:
        return False


class VRAMLogger:
    """Owns one open CSV file and the sample loop. Opening the file once and
    flushing after every row means a crash (or SIGKILL of the logger) still
    leaves every tick already on disk."""

    def __init__(self, path, pid, interval, python_exe=None,
                 rocm_timeout=10.0, torch_timeout=15.0):
        self.path = path
        self.pid = pid
        self.interval = interval
        self.python_exe = python_exe or sys.executable
        self.rocm_timeout = rocm_timeout
        self.torch_timeout = torch_timeout
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(CSV_HEADER)
        self._f.flush()
        self.samples = 0

    def write_row(self, alloc_bytes, res_bytes, rocm_free_bytes, ts=None):
        """Append one CSV row from already-sampled byte values (or None for
        unavailable sources). Exposed so the self-test can feed synthetic
        values and verify the bytes->MB conversion."""
        if ts is None:
            ts = datetime.datetime.now().isoformat(timespec="milliseconds")
        self._writer.writerow([
            ts,
            self.pid,
            _mb(alloc_bytes),
            _mb(res_bytes),
            _mb(rocm_free_bytes),
        ])
        self._f.flush()
        self.samples += 1

    def sample_once(self):
        """Run one tick: sample torch + rocm-smi, append a row. Returns the
        (alloc, res, rocm_free) tuple it just wrote (values may be None)."""
        alloc, res = sample_torch_cuda(self.python_exe, timeout=self.torch_timeout)
        rocm_free = sample_rocm_vram_free_bytes(timeout=self.rocm_timeout)
        self.write_row(alloc, res, rocm_free)
        return alloc, res, rocm_free

    def run(self):
        """Sample loop: ticks every `interval` seconds until the watched PID
        disappears, then exits. Honors KeyboardInterrupt for a clean close."""
        log(f"watching PID {self.pid}, sampling every {self.interval}s "
            f"-> {self.path}")
        try:
            while True:
                if not pid_alive(self.pid):
                    log(f"PID {self.pid} no longer exists -- exiting after "
                        f"{self.samples} sample(s).")
                    return
                try:
                    self.sample_once()
                except Exception as exc:
                    # A single bad tick must not kill the whole time-series.
                    # Log it (repo convention: no bare except: pass) and keep
                    # going -- the next tick usually recovers.
                    log(f"sample tick failed ({exc!r}) -- continuing")
                time.sleep(self.interval)
        finally:
            self.close()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pid", type=int, required=True,
                    help="Training process PID to watch. The logger exits "
                         "when this PID disappears.")
    ap.add_argument("--output", type=str, default="vram.csv",
                    help="Output CSV path (default: vram.csv). Overwritten.")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="Seconds between samples (default 5).")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python interpreter for the torch probe subprocess "
                         "(default: current interpreter). Override when the "
                         "logger runs from an env without torch but the "
                         "training env has it.")
    ap.add_argument("--rocm-timeout", type=float, default=10.0,
                    help="Timeout per rocm-smi call in seconds (default 10).")
    ap.add_argument("--torch-timeout", type=float, default=15.0,
                    help="Timeout for the torch probe subprocess in seconds "
                         "(default 15).")
    args = ap.parse_args()

    if args.interval <= 0:
        raise SystemExit("ERROR: --interval must be > 0")
    if not pid_alive(args.pid):
        raise SystemExit(
            f"ERROR: PID {args.pid} is not alive or not visible to this user."
        )

    logger = VRAMLogger(
        args.output, args.pid, args.interval, args.python,
        rocm_timeout=args.rocm_timeout, torch_timeout=args.torch_timeout,
    )
    try:
        logger.run()
    except KeyboardInterrupt:
        log("interrupted -- exiting.")
    return 0


def _self_test():
    print("[selftest] vram_log: CSV writer + rocm-smi JSON parser "
          "(no GPU required)")

    # --- rocm-smi JSON parser ---
    # Typical `rocm-smi --showmeminfo vram --json` output, two cards.
    mock = json.dumps({
        "card0": {"VRAM Free Memory (B)": 1073741824,
                  "VRAM Total Memory (B)": 17179869184},
        "card1": {"VRAM Free Memory (B)": 536870912,
                  "VRAM Total Memory (B)": 17179869184},
    })
    free = parse_rocm_smi_json(mock)
    assert free == 536870912, f"expected min free 512MB (536870912B), got {free}"
    print("  OK (parses JSON, returns MIN free across cards = 512MB)")

    # Empty / malformed / empty-object input -> None, no crash.
    assert parse_rocm_smi_json("") is None
    assert parse_rocm_smi_json("   ") is None
    assert parse_rocm_smi_json("not json at all") is None
    assert parse_rocm_smi_json("{}") is None
    assert parse_rocm_smi_json(json.dumps({"card0": {"VRAM Total Memory (B)": 1}})) is None
    print("  OK (empty/malformed/empty-object/no-free-key -> None, no crash)")

    # Field-name flexibility (rocm-smi versions rename keys).
    assert parse_rocm_smi_json(json.dumps({"card0": {"VRAM Free": 2048}})) == 2048
    print("  OK (matches any 'VRAM'+'Free' key -- version-robust)")

    # --- bytes->MB helper ---
    assert _mb(1048576) == "1"
    assert _mb(2097152) == "2"
    assert _mb(1073741824) == "1024"
    assert _mb(None) == ""
    print("  OK (bytes->MB conversion, None -> empty string)")

    # --- CSV writer (real VRAMLogger) ---
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    try:
        lg = VRAMLogger(tmp.name, pid=12345, interval=1.0)
        # Synthetic explicit row -> tests byte->MB columns + pid + header.
        lg.write_row(1048576, 2097152, 1073741824, ts="2026-07-06T12:00:00.000")
        # A real tick on this no-GPU/no-rocm-smi box: every source should
        # gracefully fall back to None and still write a well-formed row.
        alloc, res, rocm_free = lg.sample_once()
        assert alloc is None and res is None, (
            "expected None CUDA stats on a GPU-less box, got "
            f"alloc={alloc!r} res={res!r}"
        )
        lg.close()
        with open(tmp.name, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == CSV_HEADER, f"bad header: {rows[0]}"
        assert rows[1] == ["2026-07-06T12:00:00.000", "12345", "1", "2", "1024"], \
            f"bad synthetic row: {rows[1]}"
        assert len(rows[2]) == 5, f"bad fallback row width: {rows[2]}"
        assert rows[2][1] == "12345"
        assert rows[2][2] == "" and rows[2][3] == "" and rows[2][4] == "", \
            f"fallback row should have empty metric cols: {rows[2]}"
        print("  OK (header + synthetic row + graceful-fallback row all written)")
    finally:
        os.unlink(tmp.name)

    # --- PID liveness probe ---
    assert pid_alive(os.getpid()) is True, "self PID should be alive"
    # A PID far beyond any real system's max (Linux: 4194304, macOS: ~99998)
    # is guaranteed not to exist.
    assert pid_alive(2000000000) is False, "impossible PID should be dead"
    print("  OK (pid liveness: self alive, impossible pid dead)")

    print("\n[selftest] All checks passed.")


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
