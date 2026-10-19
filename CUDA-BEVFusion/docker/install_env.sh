#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Create the CUDA-BEVFusion Python environment (CUDA 11.3 + PyTorch 1.10 +
# mmcv/mmdet + nuscenes-devkit) on top of a plain nvidia/cuda base image.
#
# This mirrors CUDA-BEVFusion/bevfusion/docker/Dockerfile and is idempotent:
# when the image already provides the "bevfusion" conda env (e.g. the
# `bevfusion:latest` base), it does nothing.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if [ -x /opt/conda/envs/bevfusion/bin/python ]; then
  echo "[install_env] bevfusion conda env already present; skipping."
  exit 0
fi

apt-get update
apt-get install -y --no-install-recommends \
  wget build-essential g++ gcc libgl1-mesa-glx libglib2.0-0 \
  openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git unzip

# --- Miniconda ------------------------------------------------------------
wget --quiet \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm -f /tmp/miniconda.sh

export PATH=/opt/conda/bin:$PATH

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- bevfusion environment ------------------------------------------------
conda create -y -n bevfusion python=3.8 pip
conda clean -afy

conda install -y -n bevfusion \
  pytorch==1.10.1 \
  torchvision==0.11.2 \
  torchaudio==0.10.1 \
  cudatoolkit=11.3 \
  -c pytorch

export PATH=/opt/conda/envs/bevfusion/bin:/opt/conda/bin:$PATH

# --- Python packages ------------------------------------------------------
pip install --no-cache-dir Pillow==8.4.0
pip install --no-cache-dir tqdm
pip install --no-cache-dir torchpack
# mmcv-full ships CUDA ops.  Installing from source triggers an nvcc compile
# that fails during `docker build` (no GPU, arch autodetect fails).  Use
# OpenMMLab's prebuilt wheel index for this exact torch1.10 + cu113 combo so
# no compilation happens.  TORCH_CUDA_ARCH_LIST/FORCE_CUDA are a fallback in
# case pip still attempts a source build.
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
pip install --no-cache-dir mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install --no-cache-dir mmdet==2.20.0
pip install --no-cache-dir nuscenes-devkit
pip install --no-cache-dir mpi4py==3.0.3
pip install --no-cache-dir numba==0.48.0

# Drop the conda package cache (~8 GB after the PyTorch install), pip cache
# and temp files.  Because this script runs as a single Docker RUN, these
# files never make it into the image layer.
conda clean -afy
rm -rf /root/.cache /tmp/*
echo "[install_env] bevfusion environment ready."
