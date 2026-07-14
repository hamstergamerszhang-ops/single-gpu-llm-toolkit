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
#                 python3 train_cpt.py --model ./checkpoints/base --data ./data/train.jsonl \
#                 --save ./checkpoints/out --iters 1000 --batch 4 --lr 5e-7
#
# The --device /dev/kfd --device /dev/dri --group-add video flags give the
# container access to the AMD GPU. --shm-size matters because PyTorch's
# dataloader/multiprocessing uses /dev/shm; the default 64MB is too small for
# training and causes "Bus error" crashes on large batches.

# Base image: use the maintainer's "latest" ROCm/PyTorch tag rather than a
# hand-pinned one. A previous pass in this repo's history pinned
# "rocm6.2_ubuntu22.04_py3.10_pytorch_2.4" here -- that tag does not exist on
# Docker Hub (verified directly against the real rocm/pytorch tag list while
# reviewing this file; the real tags follow a "rocm6.2.x_ubuntuYY.MM_pyZ.W_
# pytorch_release_A.B.C" pattern, not this one), so that FROM line would have
# failed the build outright with "manifest not found". If you want a
# reproducible pin, pick an ACTUAL tag from
# https://hub.docker.com/r/rocm/pytorch/tags and verify it builds before
# committing to it -- don't hand-write a plausible-looking one.
FROM rocm/pytorch:latest

# Pin the deps this repo's scripts import. torch + ROCm come from the base
# image; install the rest explicitly. transformers is pinned to the version
# this repo's tools were actually run and tested against -- confirmed to
# register Gemma4Config's model_type correctly (see README and
# requirements.txt for how that was verified).
RUN pip install --no-cache-dir \
        safetensors \
        numpy \
        "transformers==5.7.0" \
        tensorboard \
        pytest \
        fastapi \
        uvicorn \
        pydantic

# ROCm-specific optional performance deps. These are installed against the
# ROCm stack in this base image (headers and hipcc are present). If a build
# fails, the image build fails loudly so users know the feature is unavailable
# rather than discovering it silently at runtime.

# bitsandbytes: as of current releases, the PyPI wheel ships ROCm kernels for
# CDNA archs (gfx90a, gfx942) and RDNA3 archs (gfx1100-1103). Plain
# `pip install bitsandbytes` is the recommended install path for ROCm
# (preview support per bitsandbytes' own docs). If the wheel doesn't cover
# your arch, the trainer's bnb_optimizer.py falls back to AdamW, so a missing
# build is not fatal to basic training -- but we still fail the image build
# if the install command itself errors, because "silently missing bnb" has
# been observed to OOM real runs.
RUN pip install --no-cache-dir bitsandbytes

# torchao: used for fp8 training on MI300X/MI325X. The PyPI wheel may not have
# ROCm kernels; if you need fp8 on AMD, build from source against this ROCm.
RUN pip install --no-cache-dir torchao

# flash-attn: must be built from source on ROCm, and that build is genuinely
# flaky across ROCm/PyTorch/GPU-arch combinations -- it's an optional
# perf feature (both train_cpt.py and generate.py fall back to standard
# attention with a warning if it's missing), not something the whole image
# build should die over. Keep this a soft failure, not a hard RUN.
RUN (pip install --no-cache-dir --no-build-isolation flash-attn || \
     echo "WARNING: flash-attn build failed -- --flash-attn will fall back to standard attention")

WORKDIR /work
