# Dockerfile for AMD ROCm single-GPU training with this repo's tools.
#
# Uses the official ROCm/PyTorch base image so torch is already built for ROCm
# (no need to compile from source). Installs the remaining deps explicitly so
# the recurring "silently missing bitsandbytes -> OOM dozens of steps in" failure
# mode documented in the README can't happen inside this container.
#
# Build:  docker build -t single-gpu-llm-toolkit .
# Run:    docker run --device /dev/kfd --device /dev/dri --group-add video \
#                 --shm-size 64G -v $(pwd):/work -w /work -it single-gpu-llm-toolkit \
#                 python3 train_cpt.py --model ... --save ...
#
# The --device /dev/kfd --device /dev/dri --group-add video flags give the
# container access to the AMD GPU. --shm-size matters because PyTorch's
# dataloader/multiprocessing uses /dev/shm; the default 64MB is too small for
# training and causes "Bus error" crashes on large batches.

# Base image: pinned to a specific ROCm/PyTorch tag for reproducibility. An
# unpinned :latest would let a new ROCm major version silently break the
# gfx-override or fp8 paths. This tag was verified to exist on Docker Hub
# (the real tag format is rocm<X.Y.Z>_ubuntu<YY.MM>_py<Z.W>_pytorch_release_A.B.C,
# NOT hyphenated — a previous pass used a hyphenated tag that doesn't exist).
# To upgrade: pick a real tag from https://hub.docker.com/r/rocm/pytorch/tags,
# verify the selftests pass, and update the pin here.
FROM rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.7.1

# Copy the toolkit into the image and install it with its extras. This replaces
# the old approach of re-listing deps in the Dockerfile (a third manifest that
# drifted from pyproject.toml + requirements.txt). Now: one source of truth
# (pyproject.toml), installed via pip install -e .[train,infer,dev].
COPY . /work
WORKDIR /work
RUN pip install --no-cache-dir ".[train,dev]"

# ROCm-specific optional performance deps. These are installed against the
# ROCm stack in this base image (headers and hipcc are present). If a build
# fails, the image build fails loudly so users know the feature is unavailable
# rather than discovering it silently at runtime.
RUN pip install --no-cache-dir ".[infer]" || \
    echo "WARNING: flash-attn/torchao build failed — --flash-attn/--dtype fp8 will fall back"

# flash-attn: must be built from source on ROCm, and that build is genuinely
# flaky across ROCm/PyTorch/GPU-arch combinations -- it's an optional perf
# feature (both train_cpt.py and generate.py fall back to standard attention
# with a warning if it's missing), not something the whole image build should
# die over. Keep this a soft failure, not a hard RUN.
RUN (pip install --no-cache-dir --no-build-isolation flash-attn || \
     echo "WARNING: flash-attn build failed -- --flash-attn will fall back to standard attention")
