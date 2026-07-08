#!/usr/bin/env python3
"""AMD ROCm environment bootstrap: detect the GPU's gfx architecture and set
HSA_OVERRIDE_GFX_VERSION when the detected arch isn't in PyTorch's compiled-in
list, so consumer/older AMD cards (RDNA1/2, gfx803, etc.) actually launch
kernels instead of failing silently at the first CUDA/ROCm call.

WHY THIS EXISTS
---------------
ROCm PyTorch wheels are typically compiled for a handful of gfx architectures
(e.g. gfx900, gfx906, gfx90a, gfx940, gfx941, gfx942, gfx1030, gfx1100). A
card whose architecture ISN'T in that compiled list — most consumer RDNA
cards (RX 7900 = gfx1100 usually works, but RX 6800 = gfx1030 sometimes
doesn't depending on the wheel, RX 5700 = gfx1010 almost never does, older
Fiji/Polaris = gfx803 never does) — will import torch fine but fail at the
first kernel launch with an error like "no kernel image is available for
execution on the device".

The standard fix is to set the HSA_OVERRIDE_GFX_VERSION environment variable
to a compatible arch BEFORE the ROCm device runtime initializes (see the
ORDERING NOTE below for why the auto-detect path works despite importing torch
first).

WHAT THIS DOES
--------------
  1. Probe the GPU's gfx arch via rocm-smi or /sys/class/kfd (no torch import).
  2. Import torch and read torch.cuda.get_arch_list() to see which archs this
     wheel was compiled for.
  3. If the detected arch is NOT in torch's compiled list, find the closest
     family member (same gfxNN major) that IS, and set
     HSA_OVERRIDE_GFX_VERSION to it.
  4. If no family match exists, warn loudly and DON'T override (guessing wrong
     can be worse than not overriding — a wrong arch override can produce
     silent numerical errors or crashes).
  5. Always log what it did (or didn't do).

ORDERING NOTE: HSA_OVERRIDE_GFX_VERSION must be set before the ROCm device
runtime initializes. `import torch` alone does NOT initialize the device --
but torch.cuda.is_available() does (its own docstring says it calls
cudaGetDeviceCount(), "which in turn initializes the CUDA Driver API via
cuInit()"), and ROCm's HSA runtime reads HSA_OVERRIDE_GFX_VERSION during that
same device/topology discovery step (confirmed against the real ROCR-Runtime
source: libhsakmt/src/topology.c reads the env var while building each GPU
node's properties). A previous version of this docstring claimed
get_arch_list() was side-effect-free "compiled metadata only" -- that was
wrong: torch.cuda.get_arch_list() calls is_available() internally as a gate,
so calling it BEFORE setting the env var can genuinely cause the override to
be read too late and silently not take effect on real hardware.

Fixed by having get_torch_arch_list() (below) call
torch._C._cuda_getArchFlags() directly instead of the public
get_arch_list() wrapper. That private call really is side-effect-free: the
arch list is compile-time metadata baked into the wheel, with no device or
driver interaction, so reading it doesn't trigger cuInit()/hipInit() and the
auto-detect path's ordering (detect arch -> read torch's compiled list ->
set HSA_OVERRIDE_GFX_VERSION, all before anything the trainer does actually
touches the device) is now genuinely safe, not just lazy-by-luck. If you want
a path that avoids importing torch at all before the env var is set, use
--gfx-override (the force-override path imports no torch at all).

WHAT THIS DOES NOT DO
---------------------
  - It does NOT install drivers, ROCm, or PyTorch. It assumes a working ROCm
    stack where the only issue is an arch mismatch.
  - It does NOT override if the detected arch is already in torch's list
    (the common case on MI250/MI300 — no override needed there).
  - It is NOT a guarantee that every AMD card will work — some very old archs
    (pre-gfx800) have no compatible override target in any modern wheel. This
    module surfaces that honestly rather than guessing.

Usage as a library (call BEFORE importing torch in your training script):
    from rocm_env import setup_rocm_env
    info = setup_rocm_env()          # auto-detect + override if needed
    import torch                     # safe to import now
    print(info)                      # what was set / why

Explicit override (skip auto-detection):
    setup_rocm_env(override="gfx1100")

CLI / self-test (no GPU/torch required — tests parsing + family-matching logic):
    python3 rocm_env.py --selftest
"""

import argparse
import os
import re
import subprocess

GFX_RE = re.compile(r"gfx(\d{2,4}[a-z]?)")  # matches gfx90a, gfx942, gfx803, gfx1100


def parse_kfd_gfx_target_version(ver):
    """Decode a raw `gfx_target_version` value (str or int) into a 'gfxNNN'
    arch string, or None if it can't be parsed / is a non-GPU node.

    The value can arrive as a literal 'gfxNNNN' string or in either of two
    integer encodings, so this tries decoders in order of specificity and is
    careful never to let one encoding's decoder mis-read another's value:

      1. Literal pass-through: if `ver` is a string that already contains a
         'gfxNNNN' token (rocm-smi-style), extract it with GFX_RE.

      2. Decimal-group -- the encoding AMD's own rocm_agent_enumerator
         (readFromKFD(), amd-staging branch) decodes, and what every real
         amdgpu kfd node actually writes to /sys/class/kfd/.../properties.
         The value is a PLAIN DECIMAL integer (not a packed hex word) with
         major/minor/stepping as base-10 digit GROUPS:
             major    = (ver // 10000) % 100
             minor    = (ver // 100)   % 100     (one hex digit)
             stepping =  ver % 100               (one hex digit)
             gfx      = f"gfx{major}{minor:x}{stepping:x}"
         Verified against known real values: 110000->gfx1100, 90402->gfx942,
         90010->gfx90a, 100300->gfx1030, 80003->gfx803, 120001->gfx1201.
         Real gfx IP minor/stepping are always 0..15 (single hex digits), so a
         decimal-group decode whose minor or stepping is >= 16 is NOT a real
         decimal-group value -- fall through to the bit-packed tier instead.

      3. Bit-packed fallback -- some non-sysfs sources pack the version as
         (major<<16)|(minor<<8)|stepping (major in bits 23-16, minor in bits
         15-8, stepping in bits 7-0). For those: major=(v>>16)&0xFF,
         minor=(v>>8)&0xFF, stepping=v&0xFF, and the gfx string is
         f"gfx{major}{minor:x}{stepping:x}" (stepping always emitted as one
         hex digit, 0-9 or a-f -- dropping a zero stepping would turn gfx1100
         into the bogus "gfx110"). This tier only runs when the decimal-group
         decode is implausible (minor/stepping >= 16), so it can never
         mis-decode a real (decimal-group) sysfs value.

    History note: an earlier version of this parser applied the bit-packed
    decode UNCONDITIONALLY (treating the decimal-group sysfs value as if it
    were packed), which produced silently WRONG archs for every real value
    tested (e.g. 90402 -> "gfx0111" instead of "gfx942") -- worse than not
    parsing at all, since a wrong detected arch can drive
    find_override_target() to set HSA_OVERRIDE_GFX_VERSION to a bogus value.
    The decimal-group tier is what makes real hardware decode correctly; the
    bit-packed tier is kept only as a guarded fallback, never as the primary
    decode.

    Extracted as a standalone function (rather than inlined in
    detect_gfx_arch's file-parsing loop) so it's directly unit-testable
    against known real values without needing to fake /sys/class/kfd.
    """
    # 1. Literal 'gfxNNNN' string (e.g. a rocm-smi-style value) -- pass through.
    if isinstance(ver, str):
        m = GFX_RE.search(ver)
        if m:
            return f"gfx{m.group(1)}"

    try:
        ver_int = int(ver, 16) if isinstance(ver, str) and ver.startswith("0x") else int(ver)
    except (ValueError, TypeError):
        return None
    if ver_int <= 0:
        # gfx_target_version is 0 on CPU-only KFD nodes (e.g. an APU's CPU
        # node) -- not a GPU, nothing to parse.
        return None

    # 2. Decimal-group (the real amdgpu kfd sysfs encoding).
    major = (ver_int // 10000) % 100
    minor = (ver_int // 100) % 100
    stepping = ver_int % 100
    if minor < 16 and stepping < 16:
        arch_str = f"gfx{major}{minor:x}{stepping:x}"
        m = GFX_RE.match(arch_str)
        if m:
            return f"gfx{m.group(1)}"

    # 3. Bit-packed fallback: (major<<16)|(minor<<8)|stepping. Only reached
    # when the decimal-group decode was implausible, so this can't clobber a
    # real sysfs value.
    major = (ver_int >> 16) & 0xFF
    minor = (ver_int >> 8) & 0xFF
    stepping = ver_int & 0xFF
    if major > 0 and minor < 16 and stepping < 16:
        arch_str = f"gfx{major}{minor:x}{stepping:x}"
        m = GFX_RE.match(arch_str)
        if m:
            return f"gfx{m.group(1)}"

    return None


def detect_gfx_arch():
    """Probe the GPU's gfx architecture WITHOUT importing torch (so the env
    var can be set before torch runtime init). Returns a string like 'gfx1100'
    or None if no AMD GPU / no rocm-smi / no /sys/class/kfd.

    Tries, in order:
      1. `rocm-smi --showproductname` — parse the 'gfx' string from its output.
      2. /sys/class/kfd/.../name files — each KFD node has a 'name' sysfs file
         containing the gfx arch (e.g. 'gfx1100').
    Both are read-only probes; neither modifies anything.
    """
    # 1. rocm-smi --showproductname
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode("utf-8", errors="replace")
        m = GFX_RE.search(out)
        if m:
            return f"gfx{m.group(1)}"
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass

    # 2. /sys/class/kfd/topology/nodes/*/properties — the amdkfd driver exposes
    # per-node properties files here. The `name` file in the same directory
    # contains the ASIC marketing codename (e.g. "navi10", "sienna_cichlid"),
    # NOT the gfx arch string. The gfx arch is in the `properties` file as a
    # `gfx_target_version` line; see parse_kfd_gfx_target_version() above for
    # the decode (a real bug in an earlier version of this parser silently
    # produced wrong archs from this field -- see that function's docstring).
    kfd_nodes = "/sys/class/kfd/topology/nodes"
    if os.path.isdir(kfd_nodes):
        for entry in sorted(os.listdir(kfd_nodes)):
            props_path = os.path.join(kfd_nodes, entry, "properties")
            try:
                with open(props_path) as f:
                    for line in f:
                        if line.startswith("gfx_target_version"):
                            arch = parse_kfd_gfx_target_version(line.split()[-1])
                            if arch:
                                return arch
            except (OSError, IOError):
                continue

    return None


def get_torch_arch_list():
    """Import torch and return its compiled-in arch list (e.g.
    ['sm_90', 'gfx900', 'gfx1100', ...]). Returns None if torch can't import
    or has no CUDA/ROCm build.

    Deliberately calls torch._C._cuda_getArchFlags() directly instead of the
    public torch.cuda.get_arch_list() wrapper -- this is a real correctness
    fix, not a style choice. get_arch_list() internally gates on
    is_available(), and is_available()'s own docstring says it calls
    cudaGetDeviceCount(), "which in turn initializes the CUDA Driver API via
    cuInit()" (same mechanism under ROCm's HIP/HSA translation). Checked
    against the real ROCR-Runtime source (libhsakmt/src/topology.c): HSA
    reads HSA_OVERRIDE_GFX_VERSION during topology/device discovery, which
    happens at that same driver-init step -- so calling get_arch_list() (and
    therefore is_available()) BEFORE HSA_OVERRIDE_GFX_VERSION is set can
    genuinely cause the override to be read too late and silently not take
    effect, not just "technically imprecise but harmless in practice."
    _cuda_getArchFlags() is compile-time metadata (the arch list baked into
    the wheel at build time) with no device/driver interaction at all -- it's
    the actually side-effect-free call this module needs. Falls back to the
    public get_arch_list() if the private attribute isn't there (e.g. a torch
    build where it's been renamed/removed) -- that fallback is a best-effort
    path that reintroduces the early-init risk, so it's not the primary path
    and is only reached on a torch internals change worth knowing about
    anyway (the fallback still returns valid data, it just loses the ordering
    guarantee on that one build)."""
    try:
        import torch
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_getArchFlags"):
            arch_flags = torch._C._cuda_getArchFlags()
            return arch_flags.split() if arch_flags else []
        if hasattr(torch.cuda, "get_arch_list"):
            return torch.cuda.get_arch_list()
    except Exception:
        pass
    return None


def _gfx_major(arch):
    """Extract the 'gfxNN' major prefix from a full 'gfxNNNN' string, e.g.
    'gfx1100' -> 'gfx11'. Returns None if the format is unexpected."""
    m = re.match(r"gfx(\d{2})", arch or "")
    return f"gfx{m.group(1)}" if m else None


def find_override_target(detected_arch, torch_arch_list):
    """Given a detected arch that's NOT in torch's compiled list, find the
    closest family member (same gfxNN major) that IS in the list.

    Strategy: among the archs in torch_arch_list that share the same gfxNN
    major prefix as detected_arch, pick the numerically closest one. If none
    share the major prefix, return None (no safe override — guessing across
    major families risks silent errors). If the detected arch IS already in
    the list, returns None (no override needed — the caller checks this first,
    but this is defensive).

    Handles all arch formats: 4-digit (gfx1100), 3-digit (gfx942, gfx803),
    and letter-suffix (gfx90a — MI250). The numeric comparison uses the
    digits AFTER the 2-digit major prefix; for letter-suffix archs like
    gfx90a, the letter is treated as 0 for numeric comparison.

    Returns the override arch string (e.g. 'gfx1030') or None.
    """
    if not torch_arch_list or not detected_arch:
        return None

    # If already supported, no override needed.
    def base_arch(a):
        return GFX_RE.search(a).group(0) if GFX_RE.search(a) else a
    if detected_arch in {base_arch(a) for a in torch_arch_list if GFX_RE.search(a)}:
        return None

    detected_major = _gfx_major(detected_arch)
    if not detected_major:
        return None

    # Extract the minor part (everything after the 2-digit major prefix).
    # For gfx1100 -> minor="00" -> 0; gfx1030 -> "30" -> 30; gfx942 -> "2" -> 2;
    # gfx90a -> "a" -> treated as 0 for numeric comparison.
    def extract_minor(arch):
        m = re.match(r"gfx\d{2}(.+)", arch)
        if not m:
            return None
        suffix = m.group(1)
        # Try to parse as int; if it's a letter (gfx90a), treat as 0.
        try:
            return int(suffix)
        except ValueError:
            return 0

    detected_num = extract_minor(detected_arch)
    if detected_num is None:
        return None

    candidates = []
    for a in torch_arch_list:
        if not GFX_RE.search(a):
            continue
        if _gfx_major(a) != detected_major:
            continue
        minor = extract_minor(a)
        if minor is None:
            continue
        candidates.append((a, minor))

    if not candidates:
        return None

    # Pick the numerically closest minor within the same major family.
    candidates.sort(key=lambda c: abs(c[1] - detected_num))
    return candidates[0][0]


def _set_hip_alloc_conf(conf, verbose=True):
    """Set PYTORCH_HIP_ALLOC_CONF if not already set by the user. This must
    happen BEFORE torch's caching allocator initializes (i.e. before the first
    CUDA/ROCm allocation). The env var is read once at allocator init; setting
    it after init has no effect, which is why this runs at the top of
    setup_rocm_env rather than after torch import.

    max_split_size_mb limits how large a single block the allocator will split
    for a smaller request. Without it, a large block gets fragmented into pieces
    that can't be recombined — the classic "I have 20GB free but OOM on a 2GB
    allocation" symptom on long training runs. The default 128MB is conservative
    and matches what most ROCm training guides recommend."""
    if conf is None:
        return
    if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = conf
        if verbose:
            print(f"[rocm_env] PYTORCH_HIP_ALLOC_CONF={conf} "
                  f"(prevents allocator fragmentation OOMs on long runs)")
    elif verbose:
        existing = os.environ["PYTORCH_HIP_ALLOC_CONF"]
        if existing != conf:
            print(f"[rocm_env] PYTORCH_HIP_ALLOC_CONF already set to "
                  f"'{existing}' by user — not overriding")


def setup_rocm_env(override=None, hip_alloc_conf="max_split_size_mb:128",
                   verbose=True):
    """Main entry point. Call BEFORE importing torch in a training script.

    - If `override` is given (e.g. 'gfx1100'), set HSA_OVERRIDE_GFX_VERSION to
      it directly — no detection, no torch arch-list comparison.
    - Otherwise: detect the GPU arch, import torch, get its compiled arch list,
      and override ONLY if the detected arch isn't already supported.
    - Sets PYTORCH_HIP_ALLOC_CONF to `hip_alloc_conf` (default
      'max_split_size_mb:128') if it isn't already set. This prevents the ROCm
      caching allocator from splitting large blocks into fragments that can't
      be recombined, which is the #1 cause of "phantom OOM" on long training
      runs where VRAM is actually available but fragmented. Pass None to skip.
      The default 128MB split limit is conservative; set it lower (e.g. 64) if
      you have many small allocations, or higher if you train with very large
      contiguous buffers. Set unconditionally (before GPU detection) because
      the env var must be set before the allocator initializes and is harmless
      on non-ROCm boxes (NVIDIA torch reads PYTORCH_CUDA_ALLOC_CONF instead).

    Returns a dict describing what happened:
        {'action': 'override'|'no-override'|'force-override'|'skip',
         'detected': 'gfx1100' or None,
         'torch_arch_list': [...],
         'override_value': 'gfx1030' or None,
         'hip_alloc_conf': 'max_split_size_mb:128' or None,
         'reason': '...'}
    """
    # Set PYTORCH_HIP_ALLOC_CONF early (before torch import) so the allocator
    # picks it up at init time. Set unconditionally (not gated on GPU detection)
    # because the env var must be set before the allocator initializes and is
    # harmless on non-ROCm boxes (NVIDIA torch reads PYTORCH_CUDA_ALLOC_CONF).
    _set_hip_alloc_conf(hip_alloc_conf, verbose)

    if override is not None:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = override
        info = {
            "action": "force-override",
            "detected": None,
            "torch_arch_list": None,
            "override_value": override,
            "hip_alloc_conf": hip_alloc_conf,
            "reason": f"explicit --gfx-override={override}; skipping detection",
        }
        if verbose:
            print(f"[rocm_env] HSA_OVERRIDE_GFX_VERSION={override} (forced, "
                  f"no detection)")
        return info

    detected = detect_gfx_arch()
    torch_arch_list = get_torch_arch_list()

    if detected is None:
        info = {
            "action": "skip",
            "detected": None,
            "torch_arch_list": torch_arch_list,
            "override_value": None,
            "hip_alloc_conf": hip_alloc_conf,
            "reason": "no AMD GPU detected (no rocm-smi, no /sys/class/kfd) — "
                      "not overriding",
        }
        if verbose:
            print(f"[rocm_env] no AMD GPU arch detected (rocm-smi and /sys/class/kfd "
                  f"both unavailable) — not setting HSA_OVERRIDE_GFX_VERSION. "
                  f"If you're on an AMD card, check that rocm-smi is on PATH.")
        return info

    if torch_arch_list is None:
        info = {
            "action": "skip",
            "detected": detected,
            "torch_arch_list": None,
            "override_value": None,
            "hip_alloc_conf": hip_alloc_conf,
            "reason": "torch.cuda.get_arch_list() unavailable — can't determine "
                      f"if {detected} needs an override",
        }
        if verbose:
            print(f"[rocm_env] detected {detected} but couldn't read torch's arch "
                  f"list (torch.cuda.get_arch_list() unavailable) — not overriding. "
                  f"Set HSA_OVERRIDE_GFX_VERSION manually if you hit 'no kernel "
                  f"image' errors.")
        return info

    # Normalize: torch's list may use 'gfx1100' or 'gfx1100:xnack-' variants.
    # Compare the base gfxNNNN token.
    def base_arch(a):
        return GFX_RE.search(a).group(0) if GFX_RE.search(a) else a

    torch_base = {base_arch(a) for a in torch_arch_list if GFX_RE.search(a)}

    if detected in torch_base:
        info = {
            "action": "no-override",
            "detected": detected,
            "torch_arch_list": torch_arch_list,
            "override_value": None,
            "hip_alloc_conf": hip_alloc_conf,
            "reason": f"{detected} is already in torch's compiled arch list — "
                      f"no override needed",
        }
        if verbose:
            print(f"[rocm_env] detected {detected}, already supported by this "
                  f"torch build — no override needed.")
        return info

    # Detected arch is NOT in torch's list — find a family-compatible override.
    target = find_override_target(detected, torch_arch_list)
    if target is None:
        info = {
            "action": "skip",
            "detected": detected,
            "torch_arch_list": torch_arch_list,
            "override_value": None,
            "hip_alloc_conf": hip_alloc_conf,
            "reason": f"{detected} not in torch's arch list and no same-family "
                      f"({_gfx_major(detected)}) override target found — NOT "
                      f"guessing (wrong override can cause silent errors)",
        }
        if verbose:
            print(f"[rocm_env] WARNING: detected {detected} is NOT in this torch "
                  f"build's arch list {sorted(torch_base)}, and no same-major "
                  f"({_gfx_major(detected)}) fallback exists. NOT overriding — "
                  f"a wrong cross-family override can cause silent numerical "
                  f"errors. If kernels fail with 'no kernel image', you'll need "
                  f"a torch wheel compiled for {detected}.", flush=True)
        return info

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = target
    info = {
        "action": "override",
        "detected": detected,
        "torch_arch_list": torch_arch_list,
        "override_value": target,
        "hip_alloc_conf": hip_alloc_conf,
        "reason": f"{detected} not in torch's arch list; overriding to "
                  f"{target} (same {_gfx_major(detected)} family)",
    }
    if verbose:
        print(f"[rocm_env] detected {detected} (not in torch's compiled list "
              f"{sorted(torch_base)}) — setting HSA_OVERRIDE_GFX_VERSION={target} "
              f"(closest same-family {_gfx_major(detected)} arch in the build). "
              f"This lets kernels launch on a card the wheel wasn't compiled for.")
    return info


def _self_test():
    print("[selftest] rocm_env: gfx arch detection + family-matching logic "
          "(no GPU/torch required)")

    # detect_gfx_arch() on a non-ROCm box returns None (no rocm-smi, no /sys/kfd).
    detected = detect_gfx_arch()
    print(f"  detect_gfx_arch() on this host: {detected!r} "
          f"(expected None on a non-ROCm box)")
    # Don't hard-assert None — if this runs ON a ROCm box it'd be a real arch.
    # Just assert it's a valid gfx string or None.
    assert detected is None or GFX_RE.match(detected), detected

    # parse_kfd_gfx_target_version: tries a literal pass-through, then the
    # real amdgpu kfd sysfs encoding (decimal digit groups: major*10000 +
    # minor*100 + stepping, minor/stepping as one hex digit each), then a
    # guarded bit-packed fallback. The decimal-group values below are real
    # gfx_target_version sysfs values (see rocm_agent_enumerator readFromKFD());
    # a prior UNCONDITIONAL bit-packed version of this parser mis-decoded every
    # one of them (e.g. 90402 -> "gfx0111" instead of "gfx942"), so these
    # assertions guard against re-introducing that as the primary decode.
    assert parse_kfd_gfx_target_version("110000") == "gfx1100", \
        parse_kfd_gfx_target_version("110000")
    assert parse_kfd_gfx_target_version("90402") == "gfx942", \
        parse_kfd_gfx_target_version("90402")
    assert parse_kfd_gfx_target_version("90010") == "gfx90a", \
        parse_kfd_gfx_target_version("90010")
    assert parse_kfd_gfx_target_version("100300") == "gfx1030", \
        parse_kfd_gfx_target_version("100300")
    assert parse_kfd_gfx_target_version("80003") == "gfx803", \
        parse_kfd_gfx_target_version("80003")
    assert parse_kfd_gfx_target_version("120001") == "gfx1201", \
        parse_kfd_gfx_target_version("120001")
    # Literal 'gfxNNNN' string passes straight through (tier 1).
    assert parse_kfd_gfx_target_version("gfx1100") == "gfx1100"
    assert parse_kfd_gfx_target_version("gfx90a") == "gfx90a"
    # Bit-packed fallback (tier 3): (major<<16)|(minor<<8)|stepping. These are
    # NOT real sysfs values (sysfs uses decimal-group above), but some
    # non-sysfs sources pack the version this way; the fallback handles them
    # without ever mis-decoding a real decimal-group value (each one's
    # decimal-group decode has stepping >= 16, so it falls through cleanly).
    assert parse_kfd_gfx_target_version(720896) == "gfx1100", \
        parse_kfd_gfx_target_version(720896)   # (11<<16)|(0<<8)|0
    assert parse_kfd_gfx_target_version(590850) == "gfx942", \
        parse_kfd_gfx_target_version(590850)   # (9<<16)|(4<<8)|2
    # CPU-only KFD node (gfx_target_version == 0) parses to None, not a bogus arch.
    assert parse_kfd_gfx_target_version("0") is None
    # Garbage/non-numeric input parses to None rather than raising.
    assert parse_kfd_gfx_target_version("not_a_number") is None
    assert parse_kfd_gfx_target_version(None) is None
    print("  OK (parse_kfd_gfx_target_version: literal pass-through + decimal-group "
          "real sysfs values + guarded bit-packed fallback all decode correctly)")

    # _gfx_major extracts the gfxNN prefix.
    assert _gfx_major("gfx1100") == "gfx11"
    assert _gfx_major("gfx1030") == "gfx10"
    assert _gfx_major("gfx90a") == "gfx90"  # MI250 arch, letter suffix — major is gfx90
    assert _gfx_major(None) is None
    print("  OK (_gfx_major extracts gfxNN major prefix correctly)")

    # find_override_target: same-family fallback exists.
    torch_list = ["sm_90", "gfx900", "gfx906", "gfx90a", "gfx942",
                  "gfx1030", "gfx1100"]
    # gfx1010 (RDNA1) not in list, but gfx1030 (RDNA2) is same gfx10 family.
    target = find_override_target("gfx1010", torch_list)
    assert target == "gfx1030", f"expected gfx1030, got {target}"
    # gfx1101 not in list, gfx1100 is same gfx11 family.
    target = find_override_target("gfx1101", torch_list)
    assert target == "gfx1100", f"expected gfx1100, got {target}"
    # gfx803 (Fiji/Polaris) — no gfx08 family member in this list -> None.
    target = find_override_target("gfx803", torch_list)
    assert target is None, f"expected None for gfx803 (no gfx08 family), got {target}"
    # Already-supported arch returns None (no override needed).
    target = find_override_target("gfx1100", torch_list)
    assert target is None, f"expected None for already-supported gfx1100, got {target}"
    # Empty / None lists return None.
    assert find_override_target("gfx1100", None) is None
    assert find_override_target("gfx1100", []) is None

    # Letter-suffix arch (gfx90a — MI250) and 3-digit archs (gfx942) are now
    # handled: they match same-gfx90-family candidates in the list.
    # gfx90c (not real, but tests letter-suffix override) -> closest gfx90 family.
    target = find_override_target("gfx90c", torch_list)
    assert target is not None, f"gfx90c should find a gfx90 family member, got {target}"
    assert _gfx_major(target) == "gfx90", f"override should be gfx90 family, got {_gfx_major(target)}"
    # gfx942 is already in the list -> returns None (already supported).
    target = find_override_target("gfx942", torch_list)
    assert target is None, f"gfx942 is in the list, expected None, got {target}"
    # A 3-digit arch NOT in the list but with a family member -> override.
    target = find_override_target("gfx903", ["gfx906", "gfx1100"])
    assert target == "gfx906", f"gfx903 should override to gfx906, got {target}"
    print("  OK (find_override_target handles 4-digit, 3-digit, and letter-suffix archs)")

    # setup_rocm_env with explicit override sets the env var directly.
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    info = setup_rocm_env(override="gfx1030", verbose=False)
    assert info["action"] == "force-override"
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "gfx1030"
    assert info["override_value"] == "gfx1030"
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    print("  OK (explicit override sets HSA_OVERRIDE_GFX_VERSION directly)")

    # setup_rocm_env with no override on a non-ROCm box: action='skip', doesn't
    # set the env var. (On a real ROCm box this would do real detection.)
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    info = setup_rocm_env(verbose=False)
    assert info["action"] in ("skip", "no-override", "override"), info["action"]
    if info["action"] == "skip" and "no AMD GPU" in info["reason"]:
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ, \
            "should NOT set env var when no GPU detected"
    print(f"  OK (auto-detect action={info['action']!r} on this host; env var "
          f"{'set' if 'HSA_OVERRIDE_GFX_VERSION' in os.environ else 'not set'})")
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

    print("[selftest] PYTORCH_HIP_ALLOC_CONF is set by setup_rocm_env")
    os.environ.pop("PYTORCH_HIP_ALLOC_CONF", None)
    info = setup_rocm_env(override="gfx1030", verbose=False)
    assert "PYTORCH_HIP_ALLOC_CONF" in os.environ, \
        "setup_rocm_env should set PYTORCH_HIP_ALLOC_CONF"
    assert os.environ["PYTORCH_HIP_ALLOC_CONF"] == "max_split_size_mb:128"
    os.environ.pop("PYTORCH_HIP_ALLOC_CONF", None)
    # User-set value should NOT be overridden.
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.5"
    info = setup_rocm_env(override="gfx1030", verbose=False)
    assert os.environ["PYTORCH_HIP_ALLOC_CONF"] == "garbage_collection_threshold:0.5", \
        "user-set PYTORCH_HIP_ALLOC_CONF should not be overridden"
    os.environ.pop("PYTORCH_HIP_ALLOC_CONF", None)
    # None disables it entirely.
    info = setup_rocm_env(override="gfx1030", hip_alloc_conf=None, verbose=False)
    assert "PYTORCH_HIP_ALLOC_CONF" not in os.environ, \
        "hip_alloc_conf=None should not set the env var"
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    print("  OK (sets default, respects user-set value, None disables)")

    print("\n[selftest] All checks passed (no GPU required — run on real AMD "
          "hardware to verify detection + override actually work).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION to this value (e.g. "
                         "gfx1100), skipping auto-detection.")
    ap.add_argument("--hip-alloc-conf", type=str, default="max_split_size_mb:128",
                    help="Value for PYTORCH_HIP_ALLOC_CONF (ROCm caching allocator "
                         "config). Default 'max_split_size_mb:128' prevents "
                         "fragmentation OOMs. Pass 'none' to skip.")
    args = ap.parse_args()
    if args.selftest:
        _self_test()
    else:
        conf = None if args.hip_alloc_conf.lower() == "none" else args.hip_alloc_conf
        info = setup_rocm_env(override=args.gfx_override, hip_alloc_conf=conf)
        print(f"\n{info}")


if __name__ == "__main__":
    main()
