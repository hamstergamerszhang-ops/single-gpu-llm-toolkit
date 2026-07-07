# Dockerfile for AMD ROCm single-GPU training with this repo's tools.
#
# Uses the official ROCm/PyTorch base image so torch is already built for ROCm
# (no need to compile from source). Installs the remaining deps explicitly so
# the recurring "silently missing bitsandbytes -> OOM dozens of steps in" failure
# mode documented in the README can't happen inside this container.
#
# Build:  docker build -t gemma-prune-cpt .
# Run:    docker run --device /dev/kfd --device /dev/dri --group-add video \
#                 --shm-size 64G -v $(pwd):/work -w /work -it gemma-prune-cpt \
#                 python3 train_cpt.py --model ... --save ...
#
# The --device /dev/kfd --device /dev/dri --group-add video flags give the
# container access to the AMD GPU. --shm-size matters because PyTorch's
# dataloader/multiprocessing uses /dev/shm; the default 64MB is too small for
# training and causes "Bus error" crashes on large batches.

FROM rocm/pytorch:latest

# Pin the deps this repo's scripts import. torch + ROCm come from the base
# image; install the rest explicitly. transformers is pinned to a version
# confirmed to register Gemma4Config's model_type correctly (see README).
RUN pip install --no-cache-dir \
        safetensors \
        numpy \
        "transformers==5.7.0" \
        bitsandbytes \
        tensorboard \
        torchao \
        && pip install --no-cache-dir --no-build-isolation flash-attn || \
           echo "WARNING: flash-attn build failed — --flash-attn will fall back to standard attention"

WORKDIR /work
