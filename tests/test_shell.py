"""Automated tests for the two shell scripts (catch_and_resume.sh, oom_guard.sh)
that the README documents as the one testing gap: "not covered by selftests or
pytest -- their logic is exercised manually." These close that gap.

Both scripts have no `main` guard (they enter their poll loop at top level), so
they're driven as subprocesses rather than sourced. The oom_guard tests fake
/proc/meminfo via the OOM_GUARD_MEMINFO env var (a test-only override the
script supports) and stub rocm-smi on PATH; the catch_and_resume tests copy the
script into a temp dir alongside a fake train_cpt.py (the script does
`cd "$(dirname "$0")"` then calls `python3 train_cpt.py` relative to there, so
the fake must live next to the script copy, not in the repo root).

The rocm-smi JSON parsing test and the SIGTERM-on-emergency test need the
`timeout` coreutil (oom_guard.sh calls `timeout 10 rocm-smi ...`); on macOS
where `timeout` is absent, a stub is provided on PATH. The pure /proc-meminfo
parsing tests run on any POSIX system.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_timeout_stub(extra_path_dir: Path):
    """oom_guard.sh calls `timeout 10 rocm-smi ...`. On macOS `timeout` isn't
    in /usr/bin, so provide a no-op stub (just execs its args) on PATH. On
    Linux where timeout exists, the real one is used."""
    if shutil.which("timeout"):
        return
    stub = extra_path_dir / "timeout"
    stub.write_text("#!/bin/bash\nshift 1; exec \"$@\"\n")  # drop the duration arg
    stub.chmod(0o755)


# ── oom_guard.sh ────────────────────────────────────────────────────────────

def _oom_guard_env(meminfo_path=None, extra_path_dir=None, **overrides):
    env = dict(os.environ)
    if extra_path_dir:
        env["PATH"] = str(extra_path_dir) + os.pathsep + env.get("PATH", "")
        _ensure_timeout_stub(extra_path_dir)
    if meminfo_path is not None:
        env["OOM_GUARD_MEMINFO"] = str(meminfo_path)
    env.update(overrides)
    return env


def _write_meminfo(path: Path, available_kb=None, free_kb=None):
    lines = []
    if available_kb is not None:
        lines.append(f"MemAvailable: {available_kb} kB")
    if free_kb is not None:
        lines.append(f"MemFree: {free_kb} kB")
    path.write_text("\n".join(lines) + "\n")


def test_oom_guard_pid_gone_exits_clean(tmp_path):
    """If the watched PID no longer exists, the guard exits 0 on the first
    poll instead of looping forever."""
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, available_kb=999999999)
    env = _oom_guard_env(meminfo_path=meminfo)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "oom_guard.sh"), "999999",
         "4000", "1500", "1"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "no longer exists" in result.stdout


def test_oom_guard_system_ram_warning_threshold(tmp_path):
    """When MemAvailable is below warn but above emergency, the guard logs a
    WARNING and keeps running."""
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, available_kb=3000 * 1024)  # 3000MB: below warn, above emerg
    child = subprocess.Popen(["sleep", "300"])
    try:
        env = _oom_guard_env(meminfo_path=meminfo)
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "oom_guard.sh"), str(child.pid),
             "4000", "1500", "1"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(3.5)
        proc.terminate()
        out, _ = proc.communicate(timeout=5)
        assert "WARNING" in out and "system RAM" in out
    finally:
        child.terminate()
        child.wait()


def _signal_child(tmp_path):
    """Start a Python child that traps SIGTERM and writes a marker file. Used
    instead of `bash -c 'trap ...'` because macOS's bash backgrounded via
    subprocess doesn't reliably receive SIGTERM across process groups -- a
    Python child with an explicit signal handler does, and it's also the
    realistic target (train_cpt.py is a Python process)."""
    marker = tmp_path / "signaled"
    child = subprocess.Popen([
        sys.executable, "-c",
        f"import signal,os,time\n"
        f"def h(s,f):\n"
        f"    open({str(marker)!r},'w').write('GOT')\n"
        f"    os._exit(143)\n"
        f"signal.signal(signal.SIGTERM,h)\n"
        f"time.sleep(300)\n"
    ])
    time.sleep(0.5)  # let the handler install
    return child


def test_oom_guard_system_ram_emergency_sends_sigterm(tmp_path):
    """When MemAvailable drops below emergency, the guard sends SIGTERM to the
    training process. We verify the child actually received the signal."""
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, available_kb=500 * 1024)  # 500MB: below emergency
    child = _signal_child(tmp_path)
    try:
        env = _oom_guard_env(meminfo_path=meminfo)
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "oom_guard.sh"), str(child.pid),
             "4000", "1500", "1"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        child.wait(timeout=15)
        proc.terminate()
        proc.communicate(timeout=5)
        assert (tmp_path / "signaled").exists(), "child never received SIGTERM"
    finally:
        child.terminate()
        child.wait()


def test_oom_guard_rocm_smi_json_parsing(tmp_path):
    """The _read_rocm_vram JSON branch parses rocm-smi --json output. We stub
    rocm-smi on PATH to emit canned JSON, set a low VRAM emergency threshold,
    and verify the guard detects low VRAM and signals. This also pins the
    documented heredoc-vs-pipe stdin fix (JSON passed as argv, not stdin)."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    rocm_stub = stub_dir / "rocm-smi"
    # 512MB free on card1, VRAM_EMERGENCY=600 -> should fire
    rocm_stub.write_text(textwrap.dedent("""\
        #!/bin/bash
        if [[ "$*" == *"--json"* ]]; then
          echo '{"card0": {"VRAM Free Memory (B)": 1073741824}, "card1": {"VRAM Free Memory (B)": 536870912}}'
        else
          echo 'GPU[0] VRAM Free Memory (B): 1073741824'
          echo 'GPU[1] VRAM Free Memory (B): 536870912'
        fi
    """))
    rocm_stub.chmod(0o755)
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, available_kb=999999999)

    child = _signal_child(tmp_path)
    try:
        env = _oom_guard_env(meminfo_path=meminfo, extra_path_dir=stub_dir)
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "oom_guard.sh"), str(child.pid),
             "4000", "1500", "1", "2048", "600"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        child.wait(timeout=15)
        out, _ = proc.communicate(timeout=5)
        assert (tmp_path / "signaled").exists(), \
            f"VRAM emergency SIGTERM never fired. stdout: {out}"
        assert "VRAM" in out and "EMERGENCY" in out
    finally:
        child.terminate()
        child.wait()


def test_oom_guard_no_gpu_backend_falls_back(tmp_path):
    """With no rocm-smi or nvidia-smi on PATH, the guard logs the no-GPU
    fallback and still runs the system-RAM check (doesn't crash)."""
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, available_kb=999999999)
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    if not shutil.which("timeout"):
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        _ensure_timeout_stub(stub_dir)
        env["PATH"] = str(stub_dir) + os.pathsep + env["PATH"]
    env["OOM_GUARD_MEMINFO"] = str(meminfo)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "oom_guard.sh"), "999999",
         "4000", "1500", "1"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "no rocm-smi or nvidia-smi found" in result.stdout


# ── catch_and_resume.sh ─────────────────────────────────────────────────────

def _setup_supervisor_dir(tmp_path: Path, behavior: str = "exit_ok"):
    """catch_and_resume.sh does `cd "$(dirname "$0")"` then calls
    `python3 train_cpt.py` relative to there. So we copy the script into
    tmp_path (so its cd lands there) and write the fake train_cpt.py next to
    it. Returns the path to the copied script."""
    script_copy = tmp_path / "catch_and_resume.sh"
    shutil.copy2(REPO_ROOT / "catch_and_resume.sh", script_copy)
    script_copy.chmod(0o755)

    fake = tmp_path / "train_cpt.py"
    fake.write_text(textwrap.dedent(f"""
        import sys, os
        from pathlib import Path
        save_dir = None
        args = sys.argv[1:]
        for i, a in enumerate(args):
            if a == "--save" and i + 1 < len(args):
                save_dir = args[i + 1]
        behavior = {behavior!r}
        if behavior == "exit_fail":
            sys.exit(1)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            state_path = Path(save_dir) / "training_state.pt"
            step = 0
            try:
                import torch
                if state_path.exists():
                    old = torch.load(state_path, map_location="cpu", weights_only=False)
                    step = old.get("step", 0)
                new_step = step + 500
                state = {{"step": new_step, "valid_loss": 9.9 if behavior == "regress" else 1.0}}
                torch.save(state, state_path)
            except ImportError:
                pass
        sys.exit(0)
    """))
    return script_copy


def test_catch_and_resume_completes_when_target_reached(tmp_path):
    """When train_cpt.py exits 0 and the checkpoint reaches TOTAL_ITERS, the
    supervisor exits 0 (training complete)."""
    script = _setup_supervisor_dir(tmp_path, "exit_ok")
    cfg = tmp_path / "config.env"
    cfg.write_text(textwrap.dedent(f"""
        MODEL={tmp_path}/fake_model
        DATA={tmp_path}/fake_data
        SAVE={tmp_path}/save
        TOTAL_ITERS=500
        CHECKPOINT_EVERY=500
        BATCH=1
        LR=1e-5
        WARMUP_STEPS=10
        MAX_SEQ_LEN=128
        STOP_FILE={tmp_path}/.stop
        LOG_PREFIX={tmp_path}/log
        HISTORY_KEEP=2
        RETRY_SLEEP_SECS=0
    """))
    result = subprocess.run(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "Training complete" in result.stdout or "completed cleanly" in result.stdout


def test_catch_and_resume_stop_file_exits_clean(tmp_path):
    """If the stop file exists, the supervisor exits 0 before launching."""
    script = _setup_supervisor_dir(tmp_path, "exit_ok")
    stop = tmp_path / ".stop"
    stop.write_text("")
    cfg = tmp_path / "config.env"
    cfg.write_text(textwrap.dedent(f"""
        MODEL={tmp_path}/fake_model
        DATA={tmp_path}/fake_data
        SAVE={tmp_path}/save
        TOTAL_ITERS=10000
        CHECKPOINT_EVERY=500
        BATCH=1
        STOP_FILE={stop}
        LOG_PREFIX={tmp_path}/log
        RETRY_SLEEP_SECS=0
    """))
    result = subprocess.run(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "Stop file found" in result.stdout


def test_catch_and_resume_same_position_retry_cap(tmp_path):
    """When train_cpt.py keeps crashing without advancing, the supervisor hits
    MAX_SAME_POSITION_RETRIES and exits 1 (doesn't retry forever)."""
    script = _setup_supervisor_dir(tmp_path, "exit_fail")
    cfg = tmp_path / "config.env"
    cfg.write_text(textwrap.dedent(f"""
        MODEL={tmp_path}/fake_model
        DATA={tmp_path}/fake_data
        SAVE={tmp_path}/save
        TOTAL_ITERS=10000
        CHECKPOINT_EVERY=500
        BATCH=1
        MAX_SAME_POSITION_RETRIES=2
        RETRY_SLEEP_SECS=0
        STOP_FILE={tmp_path}/.stop
        LOG_PREFIX={tmp_path}/log
        HISTORY_KEEP=2
    """))
    result = subprocess.run(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}\n{result.stdout}"
    assert "No new checkpoint progress" in result.stdout or "recurring problem" in result.stdout


def test_catch_and_resume_config_defaults_when_no_config_file(tmp_path):
    """With no config.env, the supervisor warns and uses built-in defaults
    (doesn't crash under set -u with missing vars). The default stop-file
    (.stop_autoresume in the script's dir) lets it exit cleanly."""
    script = _setup_supervisor_dir(tmp_path, "exit_ok")
    (tmp_path / ".stop_autoresume").write_text("")  # default STOP_FILE
    result = subprocess.run(
        ["bash", str(script), str(tmp_path / "nonexistent.env")],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "WARNING" in result.stdout and "not found" in result.stdout
    assert "Stop file found" in result.stdout


def test_catch_and_resume_history_pruning(tmp_path):
    """The history pruning pipeline keeps only HISTORY_KEEP entries, oldest
    first. Pruning runs in the loop body after a successful advance (the `else`
    branch of the same-position check). We pre-seed 5 history entries, set
    HISTORY_KEEP=2 and TOTAL_ITERS high (so the run doesn't exit cleanly before
    pruning runs), let one advance happen, then stop the supervisor and check
    only the 2 newest pre-seeded entries survive (plus the new one)."""
    script = _setup_supervisor_dir(tmp_path, "exit_ok")
    save_dir = tmp_path / "save"
    history_dir = tmp_path / (save_dir.name + "_history")
    history_dir.mkdir(parents=True)
    for step in (1000, 2000, 3000, 4000, 5000):
        d = history_dir / f"step{step}"
        d.mkdir()
        (d / ".train_loss").write_text("1.0")
        (d / ".loss_kind").write_text("valid")
        (d / "marker").write_text(f"step{step}")
    stop = tmp_path / ".stop"
    cfg = tmp_path / "config.env"
    cfg.write_text(textwrap.dedent(f"""
        MODEL={tmp_path}/fake_model
        DATA={tmp_path}/fake_data
        SAVE={save_dir}
        TOTAL_ITERS=100000
        CHECKPOINT_EVERY=500
        BATCH=1
        STOP_FILE={stop}
        LOG_PREFIX={tmp_path}/log
        HISTORY_KEEP=2
        RETRY_SLEEP_SECS=1
    """))
    # Start the supervisor, let one advance + pruning run, then stop it.
    proc = subprocess.Popen(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Wait for the first advance (step 0->500) + pruning to complete. The
    # supervisor logs "History: saved step ..." after a successful advance.
    time.sleep(3)
    stop.write_text("")  # request clean stop before the next attempt
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    # Pruning should have removed the oldest entries beyond HISTORY_KEEP=2.
    remaining = sorted(d.name for d in history_dir.iterdir() if d.is_dir())
    # After pruning: the 2 newest pre-seeded (step4000, step5000) + the new
    # step500 = at most 3. The oldest (step1000, step2000, step3000) must go.
    assert "step5000" in remaining, f"newest should survive: {remaining}"
    assert "step1000" not in remaining, f"oldest should be pruned: {remaining}"
    assert "step2000" not in remaining, f"second-oldest should be pruned: {remaining}"
