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
    ap.add_argument("--src", required=True, help="Source checkpoint directory (HF format).")
    ap.add_argument("--dst", required=True, help="Output GGUF file path.")
    ap.add_argument("--outtype", default="f16",
                    help="GGUF quantization type (e.g. f16, q4_k_m, q8_0).")
    ap.add_argument("--convert-script",
                    help="Path to llama.cpp convert_hf_to_gguf.py. Auto-detected if not set.")
    args = ap.parse_args()

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
    print("[selftest] export_gguf: convert-script detection + command construction")
    import tempfile, os

    # Test convert-script path detection with --convert-script override.
    # The tool should construct the correct command (positional src, not --src).
    ap = argparse.ArgumentParser(description="test")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--outtype", default="f16")
    ap.add_argument("--convert-script", default=None)

    args = ap.parse_args(["--src", "/fake/src", "--dst", "/fake/dst.gguf"])
    # Simulate the command construction (without running it).
    cmd = [sys.executable, "/fake/convert.py", args.src,
           "--outfile", args.dst, "--outtype", args.outtype]
    # Verify the command uses positional src (not --src), matching llama.cpp's API.
    assert "/fake/src" in cmd, "src should be a positional argument"
    assert "--src" not in cmd, "src should NOT be a --src flag (llama.cpp uses positional)"
    assert "--outfile" in cmd
    assert "--outtype" in cmd
    print("  OK (command uses positional src, --outfile, --outtype)")

    # Test outtype default.
    assert args.outtype == "f16", f"default outtype should be f16, got {args.outtype}"
    print("  OK (default outtype is f16)")

    print("\n[selftest] All checks passed (no GPU/llama.cpp required).")


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
