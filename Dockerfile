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

# Base image: pinned to a specific ROCm/PyTorch tag for reproducibility.
# :latest drifts and can break the hard transformers==5.7.0 pin.
# To update: browse https://hub.docker.com/r/rocm/pytorch/tags, pick a tag
# that matches your ROCm version, and verify the build before committing.
FROM rocm/pytorch:rocm6.3.4-ubuntu22.04-py3.11-pytorch-stage

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
