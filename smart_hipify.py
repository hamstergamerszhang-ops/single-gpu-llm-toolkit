#!/usr/bin/env python3
"""Smart CUDA→HIP converter — smarter than AMD's stock `hipify-perl`.

Stock `hipify-perl` does pure text substitution (cudaMalloc→hipMalloc, etc.)
which breaks on:
  - CUDA code that uses CUDA-specific headers not present in HIP
  - Kernel launch syntax differences (<<<>>> is supported but triple-chevron
    with device pointers needs device-side handling)
  - cuBLAS/cuDNN API calls that map to rocBLAS/MIOpen (different argument order)
  - CUDA-specific pragmas, attributes, built-in variables

This tool does the same text substitution as hipify-perl, but ALSO:
  - Detects which CUDA libraries are used and warns about rocBLAS/MIOpen
    equivalents that have different APIs (not drop-in replacements)
  - Flags CUDA-specific headers that have no HIP equivalent (so the user knows
    to rewrite those parts manually, rather than getting a silent compile error)
  - Adds `#include <hip/hip_runtime.h>` automatically if CUDA headers are
    detected but no HIP header is present
  - Converts `#include <cuda_*.h>` to `#include <hip/hip_*.h>` where a mapping
    exists, and WARNS (not silently skips) where it doesn't
  - Detects `__device__` / `__host__` / `__global__` qualifiers (these are
    compatible with HIP, but the tool logs how many kernels were found so the
    user can verify)
  - Provides a summary report of every substitution, flag, and warning, so the
    user can review what changed

The tool does NOT attempt to convert CUDA library calls (cuBLAS, cuDNN, cuSPARSE)
to their ROCm equivalents — those APIs differ enough that automated conversion
would produce silently wrong code. Instead it flags every such call with a
comment `/* HIPIFY: TODO — convert to rocBLAS/MIOpen manually */` and a summary
report. This is the "smart" part: honest about what can and can't be automated.

Usage:
    python3 smart_hipify.py --src kernel.cu --dst kernel.cpp
    python3 smart_hipify.py --src ./cuda_project/ --dst ./hip_project/ --recursive

Self-test (no GPU required — exercises the substitution + detection logic):
    python3 smart_hipify.py --selftest
"""

import argparse
import os
import re


def log(msg: str):
    print(f"[hipify] {msg}", flush=True)


# CUDA→HIP API name mappings (the same substitutions hipify-perl makes).
# These are the safe, drop-in replacements where the API is identical.
CUDA_TO_HIP_API = {
    # Runtime API
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemset": "hipMemset",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaSetDevice": "hipSetDevice",
    "cudaGetDevice": "hipGetDevice",
    "cudaDeviceReset": "hipDeviceReset",
    "cudaGetLastError": "hipGetLastError",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cudaStreamSynchronize": "hipStreamSynchronize",
    "cudaEventCreate": "hipEventCreate",
    "cudaEventDestroy": "hipEventDestroy",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cudaEventElapsedTime": "hipEventElapsedTime",
    "cudaMallocManaged": "hipMallocManaged",
    "cudaHostAlloc": "hipHostMalloc",
    "cudaFreeHost": "hipHostFree",
    "cudaMemcpyAsync": "hipMemcpyAsync",
    "cudaMemsetAsync": "hipMemsetAsync",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaDeviceGetAttribute": "hipDeviceGetAttribute",
    # Error handling
    "cudaSuccess": "hipSuccess",
    "cudaErrorMemoryAllocation": "hipErrorMemoryAllocation",
    "cudaErrorInvalidValue": "hipErrorInvalidValue",
    # Types
    "cudaError_t": "hipError_t",
    "cudaDeviceProp": "hipDeviceProp_t",
    "cudaStream_t": "hipStream_t",
    "cudaEvent_t": "hipEvent_t",
    "cudaMemcpyKind": "hipMemcpyKind",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    # Threading
    "cudaThreadSynchronize": "hipDeviceSynchronize",
}

# CUDA→HIP header mappings.
CUDA_TO_HIP_HEADERS = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "device_launch_parameters.h": "hip/hip_runtime.h",
    "cuda.h": "hip/hip_runtime.h",
    "cuda_api.h": "hip/hip_runtime_api.h",
}

# CUDA library calls that have NO drop-in HIP equivalent — flagged for manual
# conversion. These are the APIs where argument order / semantics differ.
CUDA_LIBRARY_CALLS = {
    # cuBLAS → rocBLAS (different handle type, different arg order)
    "cublasCreate": "rocblas_create_handle",
    "cublasDestroy": "rocblas_destroy_handle",
    "cublasSgemm": "rocblas_sgemm",
    "cublasDgemm": "rocblas_dgemm",
    "cublasGemmEx": "rocblas_gemm_ex",
    # cuDNN → MIOpen (very different API)
    "cudnnCreate": "miopenCreate",
    "cudnnDestroy": "miopenDestroy",
    "cudnnConvolutionForward": "miopenConvolutionForward",
    "cudnnBatchNormalizationForward": "miopenBatchNormalizationForward",
    # cuSPARSE → rocSPARSE
    "cusparseCreate": "rocsparse_create_handle",
    "cusparseDestroy": "rocsparse_destroy_handle",
}


def hipify_text(source: str) -> tuple:
    """Convert CUDA source text to HIP. Returns (hipified_text, report) where
    report is a dict with:
      - 'api_substitutions': list of (cuda_name, hip_name, count)
      - 'header_substitutions': list of (cuda_header, hip_header, count)
      - 'library_calls_flagged': list of (cuda_call, rocm_equivalent, count)
      - 'kernels_found': int (count of __global__ qualifiers)
      - 'hip_header_added': bool
      - 'warnings': list of str
    """
    report = {
        "api_substitutions": [],
        "header_substitutions": [],
        "library_calls_flagged": [],
        "kernels_found": 0,
        "hip_header_added": False,
        "warnings": [],
    }

    result = source

    # Count kernels (__global__ functions).
    report["kernels_found"] = len(re.findall(r"\b__global__\b", result))

    # Substitute API calls.
    for cuda_name, hip_name in CUDA_TO_HIP_API.items():
        pattern = r"\b" + re.escape(cuda_name) + r"\b"
        count = len(re.findall(pattern, result))
        if count > 0:
            result = re.sub(pattern, hip_name, result)
            report["api_substitutions"].append((cuda_name, hip_name, count))

    # Substitute headers.
    for cuda_hdr, hip_hdr in CUDA_TO_HIP_HEADERS.items():
        pattern = r'#include\s*[<"]' + re.escape(cuda_hdr) + r'[>"]'
        count = len(re.findall(pattern, result))
        if count > 0:
            result = re.sub(pattern, f'#include <{hip_hdr}>', result)
            report["header_substitutions"].append((cuda_hdr, hip_hdr, count))

    # Flag CUDA library calls that need manual conversion.
    for cuda_call, rocmm_eq in CUDA_LIBRARY_CALLS.items():
        pattern = r"\b" + re.escape(cuda_call) + r"\b"
        count = len(re.findall(pattern, result))
        if count > 0:
            # Insert a TODO comment before the first occurrence.
            todo_comment = f"/* HIPIFY: TODO — {cuda_call} → {rocmm_eq} "
            todo_comment += f"requires manual conversion (API differs) */\n"
            # Find the first occurrence's line and prepend the comment. MUST
            # use the same word-boundary `pattern` used for `count` above, not
            # a plain substring check -- a prior version did `if cuda_call in
            # line:`, which can match a longer identifier that merely
            # CONTAINS cuda_call as a substring (e.g. a wrapper function
            # `my_cublasCreate_wrapper` on an earlier line than the real
            # `cublasCreate(&handle);` call), misattaching the TODO comment
            # to the wrong line while `count` (computed via `pattern`, i.e.
            # correctly) stays accurate. Verified with a standalone repro:
            # `if cuda_call in line` attaches the comment to the wrapper
            # def's line; `pattern.search(line)` correctly skips it and
            # attaches to the real call site.
            lines = result.split("\n")
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    lines.insert(i, todo_comment.rstrip())
                    break
            result = "\n".join(lines)
            report["library_calls_flagged"].append((cuda_call, rocmm_eq, count))
            report["warnings"].append(
                f"{cuda_call} ({count}x) → {rocmm_eq}: API differs, manual "
                f"conversion required. A TODO comment was inserted."
            )

    # Add hip/hip_runtime.h if CUDA headers were found but no HIP header is
    # already present. Check against `result` (post-substitution), NOT the
    # original `source`: a CUDA header like cuda.h or cuda_runtime.h is itself
    # substituted to hip/hip_runtime.h above, so checking `source` would miss
    # that and prepend a DUPLICATE #include <hip/hip_runtime.h>.
    has_cuda = bool(report["header_substitutions"])
    has_hip = "hip/hip_runtime.h" in result
    if has_cuda and not has_hip:
        result = '#include <hip/hip_runtime.h>\n' + result
        report["hip_header_added"] = True

    # Warn about CUDA headers with no HIP mapping.
    all_includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', result)
    for inc in all_includes:
        if inc.startswith("cuda") and inc not in CUDA_TO_HIP_HEADERS:
            if inc not in [h[0] for h in report["header_substitutions"]]:
                report["warnings"].append(
                    f"#include <{inc}> has no HIP equivalent — manual rewrite needed."
                )

    return result, report


def hipify_file(src_path: str, dst_path: str, dry_run: bool = False) -> dict:
    """Hipify a single file. Returns the report dict."""
    with open(src_path, encoding="utf-8", errors="replace") as f:
        source = f.read()

    result, report = hipify_text(source)

    if not dry_run:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(result)

    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="Source .cu file or directory.")
    ap.add_argument("--dst", required=True, help="Output .cpp file or directory.")
    ap.add_argument("--recursive", action="store_true",
                    help="If --src is a directory, recursively hipify all .cu files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the conversion report, write nothing.")
    args = ap.parse_args()

    if os.path.isfile(args.src):
        # Single file.
        reports = [(args.src, args.dst, hipify_file(args.src, args.dst, args.dry_run))]
    elif os.path.isdir(args.src):
        # Directory — find .cu files.
        reports = []
        for root, dirs, files in os.walk(args.src):
            for fname in files:
                if fname.endswith(".cu") or fname.endswith(".cuh"):
                    src_f = os.path.join(root, fname)
                    rel = os.path.relpath(src_f, args.src)
                    dst_f = os.path.join(args.dst, rel)
                    # Change extension .cu -> .cpp, .cuh -> .hpp
                    if dst_f.endswith(".cu"):
                        dst_f = dst_f[:-3] + ".cpp"
                    elif dst_f.endswith(".cuh"):
                        dst_f = dst_f[:-4] + ".hpp"
                    reports.append((src_f, dst_f, hipify_file(src_f, dst_f, args.dry_run)))
            if not args.recursive:
                break
        if not reports:
            log("no .cu/.cuh files found in source directory.")
            return
    else:
        raise SystemExit(f"ERROR: {args.src} is not a file or directory.")

    # Print reports.
    total_subs = 0
    total_flags = 0
    total_warnings = 0
    for src_f, dst_f, report in reports:
        log(f"{'DRY RUN: ' if args.dry_run else ''}{src_f} -> {dst_f}")
        if report["api_substitutions"]:
            for cuda_name, hip_name, count in report["api_substitutions"]:
                log(f"  {cuda_name} -> {hip_name} ({count}x)")
            total_subs += sum(c for _, _, c in report["api_substitutions"])
        if report["header_substitutions"]:
            for cuda_hdr, hip_hdr, count in report["header_substitutions"]:
                log(f"  #include <{cuda_hdr}> -> <{hip_hdr}> ({count}x)")
        if report["library_calls_flagged"]:
            for cuda_call, rocmm_eq, count in report["library_calls_flagged"]:
                log(f"  FLAG: {cuda_call} -> {rocmm_eq} ({count}x) — manual conversion")
            total_flags += sum(c for _, _, c in report["library_calls_flagged"])
        if report["kernels_found"]:
            log(f"  kernels found (__global__): {report['kernels_found']}")
        if report["hip_header_added"]:
            log(f"  added #include <hip/hip_runtime.h>")
        for w in report["warnings"]:
            log(f"  WARNING: {w}")
            total_warnings += 1
        if not report["api_substitutions"] and not report["header_substitutions"] \
           and not report["library_calls_flagged"]:
            log(f"  (no CUDA APIs found — file may already be HIP or has no CUDA code)")

    log(f"\nsummary: {total_subs} API substitutions, {total_flags} library calls "
        f"flagged for manual conversion, {total_warnings} warnings, "
        f"{len(reports)} file(s) processed.")


def _self_test():
    print("[selftest] smart_hipify: CUDA→HIP substitution + detection logic")

    # Basic API substitution.
    src = "cudaMalloc(&ptr, size); cudaFree(ptr);"
    result, report = hipify_text(src)
    assert "hipMalloc" in result
    assert "hipFree" in result
    assert "cudaMalloc" not in result
    assert len(report["api_substitutions"]) == 2
    print("  OK (basic API substitution: cudaMalloc→hipMalloc, cudaFree→hipFree)")

    # Header substitution.
    src = '#include <cuda_runtime.h>\nint main() { return 0; }'
    result, report = hipify_text(src)
    assert "hip/hip_runtime.h" in result
    assert "cuda_runtime.h" not in result
    assert len(report["header_substitutions"]) == 1
    print("  OK (header substitution: cuda_runtime.h → hip/hip_runtime.h)")

    # HIP header not added when already present.
    src = '#include <cuda_runtime.h>\n#include <hip/hip_runtime.h>\n'
    result, report = hipify_text(src)
    assert not report["hip_header_added"]
    print("  OK (HIP header not duplicated when already present)")

    # HIP header added when CUDA headers found but no HIP header.
    # Use cuda_runtime_api.h (maps to hip/hip_runtime_api.h, NOT
    # hip/hip_runtime.h) so the auto-add path is genuinely exercised:
    # cuda_runtime.h itself substitutes to hip/hip_runtime.h, so with the
    # post-substitution duplicate check it would no longer trigger an add.
    src = '#include <cuda_runtime_api.h>\nint main() { return 0; }'
    result, report = hipify_text(src)
    assert report["hip_header_added"]
    assert result.startswith("#include <hip/hip_runtime.h>")
    print("  OK (HIP header auto-added when CUDA headers detected)")

    # No duplicate hip header when a CUDA header itself maps to
    # hip/hip_runtime.h (cuda_runtime.h -> hip/hip_runtime.h): the
    # post-substitution check must see the already-present hip header and NOT
    # prepend a second one.
    src = '#include <cuda_runtime.h>\nint main() { return 0; }'
    result, report = hipify_text(src)
    assert result.count("#include <hip/hip_runtime.h>") == 1
    assert not report["hip_header_added"]
    print("  OK (no duplicate hip header when CUDA header maps to it)")

    # Library calls flagged (not silently converted).
    src = "cublasCreate(&handle); cublasSgemm(handle, ...);"
    result, report = hipify_text(src)
    assert len(report["library_calls_flagged"]) == 2
    assert "cublasCreate" in result  # NOT substituted (flagged with TODO comment)
    assert "HIPIFY: TODO" in result
    assert any("manual conversion" in w for w in report["warnings"])
    print("  OK (library calls flagged with TODO comment, not silently converted)")

    # Kernel detection (__global__).
    src = "__global__ void my_kernel() { }"
    result, report = hipify_text(src)
    assert report["kernels_found"] == 1
    print("  OK (kernel detection: __global__ counted)")

    # No CUDA code — no substitutions.
    src = "int main() { return 0; }"
    result, report = hipify_text(src)
    assert len(report["api_substitutions"]) == 0
    assert report["kernels_found"] == 0
    print("  OK (non-CUDA file: no substitutions, no false positives)")

    # Unknown CUDA header flagged.
    src = '#include <cuda_weird_header.h>\n'
    result, report = hipify_text(src)
    assert any("cuda_weird_header.h" in w for w in report["warnings"])
    print("  OK (unknown CUDA header flagged with warning)")

    print("\n[selftest] All checks passed (no GPU required — pure text processing).")


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
