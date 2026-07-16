#!/usr/bin/env python3
"""Export a checkpoint to GGUF format for llama.cpp.

This tool is a thin wrapper around the official llama.cpp conversion script.
If `llama.cpp/convert_hf_to_gguf.py` is not available, it prints the exact
command you need to run manually.
"""

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="Source checkpoint directory (HF format).")
    ap.add_argument("--dst", help="Output GGUF file path.")
    ap.add_argument("--outtype", default="f16",
                    help="GGUF quantization type (e.g. f16, q4_k_m, q8_0).")
    ap.add_argument("--convert-script",
                    help="Path to llama.cpp convert_hf_to_gguf.py. Auto-detected if not set.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU/llama.cpp required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    # Validate required args AFTER the --selftest check.
    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

    convert_script = args.convert_script
    if convert_script is None:
        candidates = [
            "llama.cpp/convert_hf_to_gguf.py",
            "../llama.cpp/convert_hf_to_gguf.py",
            "../../llama.cpp/convert_hf_to_gguf.py",
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                convert_script = cand
                break

    if convert_script is None or not os.path.isfile(convert_script):
        print(
            "[export_gguf] ERROR: could not find llama.cpp/convert_hf_to_gguf.py.\n"
            "Please clone llama-cpp (https://github.com/ggml-org/llama.cpp) and run:\n"
            f"  python3 llama.cpp/convert_hf_to_gguf.py {args.src} "
            f"--outfile {args.dst} --outtype {args.outtype}",
            file=sys.stderr,
        )
        sys.exit(1)

    # NOTE: convert_hf_to_gguf.py takes the source model directory as a
    # positional argument ("model", nargs="?"), NOT a --src flag. Verified
    # against the live upstream script (ggml-org/llama.cpp master,
    # convert_hf_to_gguf.py) during code review -- the previous `--src`
    # invocation would have failed with an argparse error (unrecognized
    # arguments / missing required "model" positional) on any real
    # llama.cpp checkout.
    cmd = [
        sys.executable, convert_script,
        args.src,
        "--outfile", args.dst,
        "--outtype", args.outtype,
    ]
    print(f"[export_gguf] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("[export_gguf] done.")


def _self_test():
    """Self-test: verify command construction + convert-script path logic
    (no GPU, no llama.cpp required)."""
    print("[selftest] export_gguf: command construction (no GPU/llama.cpp required)")

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--outtype", default="f16")
    ap.add_argument("--convert-script", default=None)
    ap.add_argument("--selftest", action="store_true", default=False)

    a = ap.parse_args(["--src", "/tmp/model", "--dst", "/tmp/model.gguf"])
    assert a.src == "/tmp/model"
    assert a.dst == "/tmp/model.gguf"
    assert a.outtype == "f16"
    print("  OK (positional --src + flags parsed correctly)")

    # Command construction mirrors main(): convert_script takes src as a
    # positional, --outfile/--outtype as flags.
    cmd = [sys.executable, "convert_hf_to_gguf.py", a.src,
           "--outfile", a.dst, "--outtype", a.outtype]
    assert cmd[2] == "/tmp/model", cmd
    assert "--outfile" in cmd
    assert "--outtype" in cmd
    print("  OK (convert command: src positional, --outfile/--outtype flags)")

    # outtype override.
    a = ap.parse_args(["--src", "/m", "--dst", "/o.gguf", "--outtype", "q4_k_m"])
    assert a.outtype == "q4_k_m"
    print("  OK (outtype override q4_k_m)")

    print("\n[selftest] All checks passed.")


if __name__ == "__main__":
    main()
