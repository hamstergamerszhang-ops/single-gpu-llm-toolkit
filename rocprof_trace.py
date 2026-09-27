#!/usr/bin/env python3
"""Wrap a short training run with ROCm's `rocprof` profiler, then parse its
output to report AMD-specific metrics that torch.profiler does not surface
(per-kernel HIP durations, GPU utilization as kernel-time/wall-time, and a
roofline-style memory-bandwidth estimate).

WHY THIS EXISTS
---------------
torch.profiler (on ROCm, where its CUDA activity maps onto HIP) reports per-op
GPU time and a trace you can open in chrome://tracing, but it does NOT give you
the raw HIP-kernel-level breakdown that AMD's own `rocprof` tool emits — the
"which kernel ate the time" view that's the first thing an AMD engineer asks
for when a training run is slower than expected. This tool is the thinnest
possible wrapper that runs your command under `rocprof --stats --hip-trace`,
finds the `*.stats.csv` it dropped, and prints: top-K kernels by total time,
total kernel time, GPU utilization %, and a memory-bandwidth estimate — without
you having to remember rocprof's flag soup or its version-dependent output
filenames.

WHAT IT DOES
------------
  1. Locates `rocprof` (or `rocprofv2`) on PATH.
  2. Probes whether `--hip-trace` is supported; if not, falls back to
     `--stats`-only (older rocprof builds don't have `--hip-trace`).
  3. Runs:  rocprof --stats [--hip-trace] -o <output_dir>/rocprof -- <command>
  4. Globs <output_dir>/*.csv, finds the *stats* CSV, parses it, and prints a
     report: top-K kernels, total kernel time, GPU utilization %, and a
     memory-bandwidth estimate.
  5. If rocprof isn't installed, falls back to torch.profiler (see below) with
     a clear warning.

REQUIREMENTS / HONEST CAVEATS
-----------------------------
  - This tool REQUIRES ROCm's `rocprof` binary for its primary path. It is part
    of the ROCm stack (package `rocprofiler-dev` / `rocprof` depending on
    distro). If you don't have it, you get the torch.profiler fallback, which
    is a different (and coarser) view — see below. There is no way to get the
    AMD-kernel-level stats this tool reports without rocprof; that's the whole
    reason this tool exists instead of just using torch.profiler.
  - GPU utilization here is "sum of per-kernel TotalDuration / wall-clock time
    of the wrapped run," reported as a percentage. On a single stream this is a
    fair serial-utilization number; with concurrent kernels across multiple
    HIP streams it can UNDER-count (overlapping kernels' durations sum but
    overlap in wall-time) — i.e. it's a lower bound on busy-time, not a true
    SM-occupancy metric. For true occupancy, profile with rocprof's occupancy
    counters or use omniperf.
  - The memory-bandwidth estimate is a ROOFLINE HEURISTIC, NOT a measurement:
    `--stats` reports kernel durations only, never bytes moved, so true
    achieved bandwidth is not derivable from this output. We report
    `gpu_util_fraction * peak_HBM_bandwidth` as a "if this workload were
    memory-bound, it could be moving at most this much" upper bound, with
    peak HBM BW configurable (--hbm-bandwidth-gbs, default 5300 GB/s for
    MI300X). For a MEASURED bandwidth, re-run with rocprof's memory/PMC
    counters (e.g. `rocprof --stats -c GRBM_COUNT,TCC_* ...`) or use omniperf.
  - rocprof's stats CSV column names vary across ROCm versions (e.g.
    `TotalDuration` vs `TotalDuration(ns)` vs `Total`). The parser matches
    column headers by substring, not exact name, so it survives the rename
    churn — but if a future rocprof restructures the file entirely (e.g. JSON),
    the parser will need updating. The glob is permissive (`*.csv`) so a
    renamed output file is still found.

TORCH.PROFILER FALLBACK
-----------------------
When rocprof is absent and the wrapped command is a Python invocation, this
tool writes a tiny launcher into <output_dir> that wraps the target script in
`torch.profiler.profile(activities=[CPU, CUDA])` via runpy, then runs it. On
ROCm, torch.profiler's "CUDA" activity maps to HIP, so you still get a per-op
GPU-time table — just aggregated at the torch-op level, not the HIP-kernel
level rocprof gives you. Two honest limitations of this fallback: (1) it
profiles the WHOLE script including model load + tokenizer init, which
dilutes the per-step signal — keep --iters small; (2) it can only wrap a
Python script (a non-Python command falls back to plain subprocess +
wall-clock timing, no profiler). For real AMD kernel metrics, install rocprof.

Usage:
    python3 rocprof_trace.py --output ./rocprof_out \\
        -- python3 train_cpt.py --model ./checkpoints/base --iters 5

    # Force --stats only (skip --hip-trace, e.g. if it crashes your run):
    python3 rocprof_trace.py --output ./rocprof_out --no-hip-trace \\
        -- python3 train_cpt.py --iters 5

Self-test (no GPU / no rocprof / no torch required — exercises the stats-CSV
parser, column-flexibility, kernel sorting, and report formatting with a tiny
fake CSV):
    python3 rocprof_trace.py --selftest
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str):
    print(f"[rocprof_trace] {msg}", flush=True)


# ── rocprof discovery + capability probe ─────────────────────────────────────

def find_rocprof() -> str | None:
    """Return the path to `rocprof` or `rocprofv2` on PATH, or None.

    `rocprof` is the canonical name; `rocprofv2` is the newer ROCm 5.x+ binary
    that supports the same flags we use. We prefer `rocprof` (broader
    availability) then `rocprofv2`."""
    for name in ("rocprof", "rocprofv2"):
        path = shutil.which(name)
        if path:
            return path
    return None


def rocprof_supports_hip_trace(rocprof_path: str) -> bool:
    """Probe `rocprof --help` for a `--hip-trace` flag. Older rocprof builds
    don't have it and will error out at run time if passed it; probing first
    avoids a wasted failed run. Returns True if the help text mentions
    --hip-trace. Conservative: if the help call itself fails (non-zero exit,
    can't parse), returns False so the caller falls back to --stats-only."""
    try:
        proc = subprocess.run(
            [rocprof_path, "--help"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Match the flag as a word so "--hip-trace" matches but "--hip-trace-foo"
    # or a substring inside another word doesn't false-positive.
    return "--hip-trace" in text


def build_rocprof_command(rocprof_path: str, output_dir: Path,
                          command: list[str], use_hip_trace: bool) -> list[str]:
    """Assemble the rocprof argv. Output goes to <output_dir>/rocprof as a
    prefix; rocprof appends .stats.csv (and, with --hip-trace, the trace CSVs)
    to that prefix. The wrapped command is appended verbatim as trailing args
    (the classic rocprof form — no `--` separator — which works across the
    widest range of rocprof versions)."""
    prefix = str(output_dir / "rocprof")
    argv = [rocprof_path, "--stats"]
    if use_hip_trace:
        argv.append("--hip-trace")
    argv += ["-o", prefix]
    argv += command
    return argv


# ── stats CSV parsing (column-name-flexible, version-robust) ─────────────────

def find_columns(header: list[str]) -> tuple[str | None, str | None, str | None]:
    """Given a CSV header row, return (name_col, total_time_col, calls_col)
    by matching column names by substring across rocprof's version-dependent
    naming. Returns (None, None, None) if no recognizable name + time columns
    are found (caller treats the file as unparseable).

    Matching rules (deliberately loose — rocprof renames columns between
    versions, e.g. TotalDuration vs TotalDuration(ns) vs Total):
      - name:    first column whose header contains 'name' or 'kernel'
      - total:   first column whose header contains 'total' AND ('duration'
                 or 'ns' or 'time'); falls back to any header containing
                 'total'; falls back to any header containing 'duration'
      - calls:   first column whose header contains 'call' or 'num'
    Matching is case-insensitive."""
    name_col = total_col = calls_col = None
    for col in header:
        low = col.lower()
        if name_col is None and ("name" in low or "kernel" in low):
            name_col = col
        if total_col is None and "total" in low and \
                ("duration" in low or "ns" in low or "time" in low):
            total_col = col
        if calls_col is None and ("call" in low or "num" in low):
            calls_col = col
    # Fallbacks for total if the strict match missed.
    if total_col is None:
        for col in header:
            if "total" in col.lower():
                total_col = col
                break
    if total_col is None:
        for col in header:
            if "duration" in col.lower():
                total_col = col
                break
    return name_col, total_col, calls_col


def parse_duration_ns(value: str) -> float:
    """Parse a rocprof duration cell to nanoseconds as a float. rocprof stats
    durations are in nanoseconds, but cells may carry a unit suffix
    ('ns'/'us'/'ms'/'s'), commas, or be wrapped in quotes. Handles all of
    those; returns 0.0 for an unparseable cell (rather than aborting the whole
    parse — a single bad cell shouldn't sink the report)."""
    if value is None:
        return 0.0
    s = str(value).strip().strip('"').strip("'").replace(",", "")
    if not s:
        return 0.0
    # Strip a trailing unit if present (case-insensitive).
    unit = None
    for u in ("ns", "us", "ms", "s"):
        if s.lower().endswith(u):
            unit = u
            s = s[: -len(u)].strip()
            break
    try:
        n = float(s)
    except ValueError:
        return 0.0
    # Convert to ns based on the stripped unit. rocprof stats are normally
    # already in ns (unit absent or 'ns'), but handle the others defensively.
    if unit == "us":
        n *= 1e3
    elif unit == "ms":
        n *= 1e6
    elif unit == "s":
        n *= 1e9
    return n


def parse_stats_csv(path: Path) -> list[dict]:
    """Parse a rocprof stats CSV into a list of kernel dicts, sorted by total
    time DESCENDING. Each dict: {name, total_ns, calls}. Rows that are clearly
    summary/aggregate lines (name == 'Total' or empty) are skipped. Returns []
    if the file can't be parsed (no recognizable columns) — caller reports
    "no parseable stats found" rather than crashing.

    The file may have non-CSV preamble lines before the header (some rocprof
    versions emit a 'Statistics' banner). We scan for the header row (the row
    containing a name-like column) and parse from there.

    Returns [] if the file doesn't exist or can't be opened (defensive — the
    real flow only calls this on a file found by find_stats_csv, which already
    checked existence, but a missing file shouldn't crash a caller)."""
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            raw_rows = list(csv.reader(f))
    except (FileNotFoundError, OSError):
        return []

    # Find the header row: the first row whose cells include a name-like column.
    header_idx = None
    for i, row in enumerate(raw_rows):
        if find_columns(row)[0] is not None and find_columns(row)[1] is not None:
            header_idx = i
            break
    if header_idx is None:
        return []

    header = raw_rows[header_idx]
    name_col, total_col, calls_col = find_columns(header)
    if name_col is None or total_col is None:
        return []

    name_idx = header.index(name_col)
    total_idx = header.index(total_col)
    calls_idx = header.index(calls_col) if calls_col is not None else None

    kernels = []
    for row in raw_rows[header_idx + 1:]:
        if not row or len(row) <= max(name_idx, total_idx):
            continue
        name = row[name_idx].strip()
        if not name or name.lower() in ("total", "summary"):
            continue
        total_ns = parse_duration_ns(row[total_idx])
        calls = None
        if calls_idx is not None and calls_idx < len(row):
            try:
                calls = int(float(row[calls_idx].strip().replace(",", "")))
            except (ValueError, AttributeError):
                calls = None
        kernels.append({"name": name, "total_ns": total_ns, "calls": calls})

    # Sort by total time descending — the whole point of the report.
    kernels.sort(key=lambda k: k["total_ns"], reverse=True)
    return kernels


def find_stats_csv(output_dir: Path) -> Path | None:
    """Glob <output_dir> for the stats CSV. Prefers a file whose name contains
    'stats'; falls back to any .csv. Returns None if no CSV exists (rocprof
    may not have written one if the wrapped command failed before launching any
    kernels)."""
    if not output_dir.exists():
        return None
    csvs = sorted(output_dir.glob("*.csv"))
    if not csvs:
        return None
    for c in csvs:
        if "stats" in c.name.lower():
            return c
    return csvs[0]


# ── report formatting ────────────────────────────────────────────────────────

def compute_metrics(kernels: list[dict], wall_time_s: float,
                    hbm_peak_gbs: float) -> dict:
    """Compute the reported metrics from the parsed kernel list + wall-clock
    time of the wrapped run. Returns a dict with:
      - total_kernel_time_s
      - gpu_util_pct  (kernel_time / wall_time; lower bound on busy time)
      - mem_bw_gbs    (roofline heuristic: util_frac * peak HBM BW — NOT measured)
      - mem_bw_note   (the honesty caveat string for the report)
      - n_kernels
    See module docstring for the GPU-util and bandwidth caveats."""
    total_ns = sum(k["total_ns"] for k in kernels)
    total_s = total_ns / 1e9
    wall_ns = wall_time_s * 1e9 if wall_time_s > 0 else 0.0
    util_frac = (total_ns / wall_ns) if wall_ns > 0 else 0.0
    # Clamp the reported util to [0,1] for the bandwidth heuristic only (a
    # value >1 from overlapping kernels would inflate the heuristic past peak,
    # which is nonsensical for a "max achievable" bound). The raw util is still
    # reported un-clamped below so concurrent-overlap under-counting is visible.
    bw_frac = min(util_frac, 1.0)
    return {
        "total_kernel_time_s": total_s,
        "gpu_util_pct": util_frac * 100.0,
        "mem_bw_gbs": bw_frac * hbm_peak_gbs,
        "mem_bw_note": ("roofline heuristic = gpu_util_frac * peak HBM BW "
                        "(NOT a measurement — --stats has no byte counts)"),
        "n_kernels": len(kernels),
    }


def format_report(kernels: list[dict], metrics: dict, top_k: int,
                  stats_csv: Path | None, wall_time_s: float) -> str:
    """Format the final human-readable report string."""
    lines = []
    lines.append("=" * 78)
    lines.append("rocprof trace report")
    lines.append("=" * 78)
    lines.append(f"stats csv:        {stats_csv if stats_csv else '(none found)'}")
    lines.append(f"wall-clock:       {wall_time_s:.3f} s")
    lines.append(f"kernels parsed:   {metrics['n_kernels']}")
    lines.append(f"total kernel time:{metrics['total_kernel_time_s']:.3f} s")
    lines.append(f"GPU utilization:  {metrics['gpu_util_pct']:.1f}%  "
                 f"(kernel_time / wall_time; lower bound — see caveats)")
    lines.append(f"mem bandwidth:    {metrics['mem_bw_gbs']:.1f} GB/s  "
                 f"[{metrics['mem_bw_note']}]")
    lines.append("")
    lines.append(f"top-{top_k} kernels by total time:")
    lines.append("-" * 78)
    header = f"{'#':>3}  {'total (ms)':>12}  {'calls':>10}  {'avg (us)':>12}  kernel"
    lines.append(header)
    lines.append("-" * 78)
    top = kernels[:top_k]
    for i, k in enumerate(top, 1):
        total_ms = k["total_ns"] / 1e6
        calls = k["calls"]
        calls_str = f"{calls}" if calls is not None else "n/a"
        avg_us = (k["total_ns"] / calls / 1e3) if calls else float("nan")
        avg_str = f"{avg_us:.2f}" if calls else "n/a"
        # Truncate very long kernel names so the table stays readable.
        name = k["name"]
        if len(name) > 60:
            name = name[:57] + "..."
        lines.append(f"{i:>3}  {total_ms:>12.3f}  {calls_str:>10}  "
                     f"{avg_str:>12}  {name}")
    if not top:
        lines.append("(no kernels parsed — rocprof may not have written a "
                     "stats CSV, or its format wasn't recognized)")
    lines.append("=" * 78)
    return "\n".join(lines)


# ── primary path: run under rocprof ──────────────────────────────────────────

def run_under_rocprof(rocprof_path: str, command: list[str], output_dir: Path,
                      top_k: int, hbm_peak_gbs: float,
                      force_no_hip_trace: bool) -> int:
    """Run `command` under rocprof, then parse + report. Returns the wrapped
    command's exit code (so CI can fail on a training crash, not just on a
    profiling hiccup)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    use_hip_trace = False if force_no_hip_trace else \
        rocprof_supports_hip_trace(rocprof_path)
    if force_no_hip_trace:
        log("--no-hip-trace set: using --stats only")
    elif use_hip_trace:
        log("rocprof supports --hip-trace: using --stats --hip-trace")
    else:
        log("rocprof does not advertise --hip-trace: falling back to --stats only")

    argv = build_rocprof_command(rocprof_path, output_dir, command, use_hip_trace)
    log(f"running: {' '.join(argv)}")

    start = time.time()
    proc = subprocess.run(argv)
    wall_time_s = time.time() - start

    if proc.returncode != 0:
        log(f"WARNING: wrapped command exited {proc.returncode} — parsing "
            f"whatever rocprof wrote anyway")

    stats_csv = find_stats_csv(output_dir)
    if stats_csv is None:
        log(f"ERROR: no .csv found under {output_dir} — rocprof may not have "
            f"run any kernels (did the wrapped command actually launch GPU "
            f"work?) or wrote elsewhere. Check {output_dir}.")
        # Still report wall-clock so the user gets *something*.
        print(format_report([], compute_metrics([], wall_time_s, hbm_peak_gbs),
                            top_k, stats_csv, wall_time_s))
        return proc.returncode

    log(f"parsing {stats_csv}")
    kernels = parse_stats_csv(stats_csv)
    if not kernels:
        log(f"WARNING: parsed 0 kernels from {stats_csv} — the CSV format may "
            f"be unrecognized (rocprof renamed columns again?). Raw file left "
            f"in place for manual inspection.")
    metrics = compute_metrics(kernels, wall_time_s, hbm_peak_gbs)
    print(format_report(kernels, metrics, top_k, stats_csv, wall_time_s))
    return proc.returncode


# ── fallback path: torch.profiler (Python-only, honest limitations) ──────────

_TORCH_PROFILER_LAUNCHER = '''\
import runpy
import sys

target = sys.argv[1]
sys.argv = sys.argv[1:]  # so the target script sees [script, *args] as argv

import torch
from torch.profiler import profile, ProfilerActivity

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)  # maps to HIP on ROCm

with profile(activities=activities) as prof:
    runpy.run_path(target, run_name="__main__")

print()
print("=" * 78)
print("torch.profiler report (CUDA activity = HIP on ROCm)")
print("=" * 78)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
'''


def run_with_torch_profiler(command: list[str], output_dir: Path,
                            top_k: int) -> int:
    """Fallback when rocprof is unavailable. Writes a torch.profiler launcher
    into <output_dir> and runs the target Python script under it. See module
    docstring for the two honest limitations: profiles the whole script
    (including load/init), and Python-only. For a non-Python command, falls
    back to a plain timed subprocess (no profiler) — there's no way to use
    torch.profiler on a non-Python process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = output_dir / "_torch_profiler_launcher.py"
    launcher.write_text(_TORCH_PROFILER_LAUNCHER, encoding="utf-8")

    # Only a Python invocation can be wrapped by torch.profiler via runpy.
    if not command or not os.path.basename(command[0]).lower().startswith("python"):
        log(f"WARNING: torch.profiler fallback can only wrap a Python script; "
            f"command {command[0]!r} is not python — running it plainly with "
            f"wall-clock timing only (no profiler).")
        start = time.time()
        proc = subprocess.run(command)
        wall = time.time() - start
        print(format_report([], compute_metrics([], wall, 0.0), top_k,
                            None, wall))
        print("\n[rocprof_trace] NOTE: install rocprof for AMD kernel-level "
              "metrics; torch.profiler cannot instrument this non-Python command.")
        return proc.returncode

    # Strip the leading python interpreter; the launcher IS the interpreter now.
    target_script = command[1]
    target_args = command[2:]
    argv = [sys.executable, str(launcher), target_script] + target_args

    log(f"WARNING: rocprof not found — running under torch.profiler instead. "
        f"This gives torch-op-level GPU time (CUDA activity = HIP on ROCm), "
        f"NOT the HIP-kernel-level breakdown rocprof provides. Keep --iters "
        f"small: the launcher profiles the WHOLE script including model load.")
    log(f"running: {' '.join(argv)}")
    start = time.time()
    proc = subprocess.run(argv)
    wall = time.time() - start
    print(f"\n[rocprof_trace] wrapped run wall-clock: {wall:.3f} s "
          f"(exit {proc.returncode})")
    print("[rocprof_trace] For AMD kernel-level metrics (top kernels, GPU "
          "util, bandwidth), install rocprof (ROCm's `rocprof` binary) and "
          "re-run — torch.profiler cannot replace it for that view.")
    return proc.returncode


# ── self-test (no GPU / no rocprof / no torch) ───────────────────────────────

def self_test():
    import tempfile

    print("[selftest] column finder: matches name/total/calls across rocprof "
          "version-dependent header names")
    name, total, calls = find_columns(
        ["Name", "Calls", "TotalDuration", "Average", "Min", "Max", "Percentage"])
    assert name == "Name" and total == "TotalDuration" and calls == "Calls", \
        (name, total, calls)
    # Alternate naming (ns suffix, 'Kernel' for name, 'Num' for calls).
    name, total, calls = find_columns(
        ["Kernel", "NumCalls", "TotalDuration(ns)", "Average(ns)"])
    assert name == "Kernel" and total == "TotalDuration(ns)" and \
        calls == "NumCalls", (name, total, calls)
    # Bare 'Total' fallback when no 'duration'/'time' present.
    name, total, calls = find_columns(["name", "total", "count"])
    assert name == "name" and total == "total", (name, total)
    # No recognizable columns -> all None.
    name, total, calls = find_columns(["foo", "bar", "baz"])
    assert name is None and total is None, (name, total)
    print("  OK (handles TotalDuration, TotalDuration(ns), bare Total, "
          "Kernel/NumCalls, and missing columns)")

    print("[selftest] duration parser: strips units/commas/quotes, converts to ns")
    assert parse_duration_ns("12345") == 12345.0
    assert parse_duration_ns("12,345") == 12345.0
    assert parse_duration_ns('"12,345"') == 12345.0
    assert parse_duration_ns("5us") == 5000.0
    assert parse_duration_ns("3ms") == 3_000_000.0
    assert parse_duration_ns("2s") == 2_000_000_000.0
    assert parse_duration_ns("") == 0.0
    assert parse_duration_ns("not_a_number") == 0.0
    assert parse_duration_ns(None) == 0.0
    print("  OK (plain, comma'd, quoted, us/ms/s units, and garbage all handled)")

    print("[selftest] stats CSV parser: sorts by total time, skips Total/empty")
    fake_csv = (
        "Statistics for rocprof\n"          # preamble banner — must be skipped
        "Name,Calls,TotalDuration,Percentage\n"
        "kernel_A,100,5000000,50.0\n"
        "kernel_B,50,9000000,40.0\n"        # B has the largest total -> sorts first
        ",0,0,0.0\n"                        # empty name — skip
        "Total,150,14000000,100.0\n"        # aggregate row — skip
        "kernel_C,10,1000000,10.0\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "rocprof.stats.csv"
        p.write_text(fake_csv, encoding="utf-8")
        kernels = parse_stats_csv(p)
    assert [k["name"] for k in kernels] == ["kernel_B", "kernel_A", "kernel_C"], \
        [k["name"] for k in kernels]
    assert kernels[0]["total_ns"] == 9_000_000.0
    assert kernels[0]["calls"] == 50
    assert kernels[2]["calls"] == 10
    print("  OK (order B,A,C; Total/empty rows skipped; calls parsed)")

    print("[selftest] alternate-header CSV parses with substring matching")
    alt_csv = (
        "Kernel,NumCalls,TotalDuration(ns)\n"
        "long::kernel::name,7,7770000\n"
        "short,3,1000\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "out.csv"
        p.write_text(alt_csv, encoding="utf-8")
        kernels = parse_stats_csv(p)
    assert len(kernels) == 2 and kernels[0]["name"] == "long::kernel::name", kernels
    assert kernels[0]["total_ns"] == 7_770_000.0 and kernels[0]["calls"] == 7
    print("  OK (Kernel/NumCalls/TotalDuration(ns) headers recognized)")

    print("[selftest] metrics: util = kernel_time/wall_time; bandwidth heuristic")
    metrics = compute_metrics(kernels, wall_time_s=0.01,
                              hbm_peak_gbs=5300.0)
    # total kernel time = 7.770e6 + 1e3 ns = 0.007771 s; wall = 0.01 s
    assert abs(metrics["total_kernel_time_s"] - 0.007771) < 1e-6, metrics
    assert abs(metrics["gpu_util_pct"] - 77.71) < 1e-2, metrics
    # bw heuristic = util_frac(<=1) * 5300 = 0.7771 * 5300
    assert abs(metrics["mem_bw_gbs"] - 0.7771 * 5300) < 1e-2, metrics
    assert "NOT a measurement" in metrics["mem_bw_note"]
    print(f"  OK (util={metrics['gpu_util_pct']:.1f}% "
          f"bw={metrics['mem_bw_gbs']:.1f} GB/s heuristic)")

    print("[selftest] bandwidth heuristic clamps util>1 (concurrent kernels) "
          "to peak, but reports raw util un-clamped")
    metrics = compute_metrics([{"total_ns": 2e9}], wall_time_s=1.0,
                              hbm_peak_gbs=5300.0)
    # 2s kernel time / 1s wall -> 200% raw util, but bw clamps to 1.0*peak.
    assert abs(metrics["gpu_util_pct"] - 200.0) < 1e-6, metrics
    assert metrics["mem_bw_gbs"] == 5300.0, metrics
    print("  OK (raw util=200% reported; bw heuristic clamped to peak 5300 GB/s)")

    print("[selftest] report formatting: produces aligned top-K table, omits "
          "rows beyond top-K")
    # `kernels` here has 2 entries (long::kernel::name, short). top_k=1 shows
    # only the top one and omits 'short' — verifies the row limit is applied.
    report = format_report(kernels, metrics, top_k=1, stats_csv=Path("x.csv"),
                           wall_time_s=1.0)
    assert "top-1 kernels" in report
    assert "long::kernel::name" in report
    assert "short" not in report  # only top-1 shown; 'short' is 2nd -> omitted
    assert "GPU utilization" in report
    assert "mem bandwidth" in report
    assert "GB/s" in report
    print("  OK (table header + top-1 row + metrics present; row 2 omitted)")

    print("[selftest] empty/missing CSV handled gracefully")
    assert parse_stats_csv(Path("/nonexistent/path.csv")) == []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        empty = td / "empty.csv"
        empty.write_text("no,recognizable,header\n1,2,3\n", encoding="utf-8")
        assert parse_stats_csv(empty) == []
    print("  OK (missing file -> []; headerless CSV -> [])")

    print("[selftest] rocprof command assembly (no rocprof needed)")
    cmd = build_rocprof_command("/fake/rocprof", Path("/fake/out"),
                                ["python3", "train_cpt.py", "--iters", "5"],
                                use_hip_trace=True)
    assert cmd[0] == "/fake/rocprof", cmd
    assert "--stats" in cmd and "--hip-trace" in cmd, cmd
    assert "-o" in cmd, cmd
    assert cmd[cmd.index("-o") + 1] == "/fake/out/rocprof", cmd
    assert cmd[-4:] == ["python3", "train_cpt.py", "--iters", "5"], cmd
    # stats-only (no hip-trace) variant omits --hip-trace.
    cmd2 = build_rocprof_command("/fake/rocprof", Path("/fake/out"),
                                 ["python3", "x.py"], use_hip_trace=False)
    assert "--hip-trace" not in cmd2 and "--stats" in cmd2, cmd2
    print("  OK (argv order: rocprof --stats --hip-trace -o prefix <cmd>; "
          "stats-only omits --hip-trace)")

    print("[selftest] find_stats_csv prefers *stats* over other CSVs")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "rocprof.csv").write_text("a,b\n1,2", encoding="utf-8")
        (td / "rocprof.stats.csv").write_text("Name,Total\nk,1", encoding="utf-8")
        found = find_stats_csv(td)
        assert found is not None and found.name == "rocprof.stats.csv", found
        # Falls back to any csv if no 'stats' file.
        (td / "rocprof.stats.csv").unlink()
        found = find_stats_csv(td)
        assert found is not None and found.name == "rocprof.csv", found
        # None if dir empty.
        (td / "rocprof.csv").unlink()
        assert find_stats_csv(td) is None
    print("  OK (prefers *stats*; falls back to any .csv; None if empty)")

    print("\n[selftest] All checks passed (no GPU/rocprof/torch required — run "
          "with a real rocprof + training command on AMD hardware for actual "
          "metrics).")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory for rocprof artifacts + report. "
                         "Required. Created if missing.")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top kernels to list in the report "
                         "(default 10).")
    ap.add_argument("--no-hip-trace", action="store_true", default=False,
                    help="Force --stats only (skip --hip-trace). Use if "
                         "--hip-trace crashes the wrapped run or produces "
                         "huge trace files you don't need.")
    ap.add_argument("--hbm-bandwidth-gbs", type=float, default=5300.0,
                    help="Peak HBM bandwidth in GB/s for the memory-bandwidth "
                         "roofline heuristic (default 5300 = MI300X). Only "
                         "affects the heuristic estimate, which is NOT a "
                         "measurement — see docstring.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU/rocprof/torch "
                         "required) and exit.")
    # The wrapped command, after a `--` separator. REMAINDER captures the `--`
    # token itself; we strip it in main().
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="Command to profile, after `--` "
                         "(e.g. -- python3 train_cpt.py --iters 5).")
    args = ap.parse_args()

    if args.selftest:
        self_test()
        return

    if args.output is None:
        ap.error("--output is required (unless --selftest).")

    # Strip a leading `--` that argparse REMAINDER captures verbatim.
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("a command to profile is required after `--` "
                 "(e.g. -- python3 train_cpt.py --iters 5).")

    rocprof = find_rocprof()
    if rocprof is None:
        log("WARNING: rocprof not found on PATH — falling back to "
            "torch.profiler (Python-only, coarser; see docstring).")
        rc = run_with_torch_profiler(command, args.output, args.top_k)
        sys.exit(rc)

    log(f"using rocprof: {rocprof}")
    rc = run_under_rocprof(rocprof, command, args.output, args.top_k,
                           args.hbm_bandwidth_gbs, args.no_hip_trace)
    sys.exit(rc)


if __name__ == "__main__":
    main()
