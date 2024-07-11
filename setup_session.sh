#!/bin/bash

# Request resources
salloc --cpus-per-gpu=2 --gpus=2 --time=2:30:00 --partition=gpu_devel

# Load the required modules
module load StdEnv
module load miniconda/24.3.0
module load CUDA/12.1.1
module load GCCcore/12.2.0
module load GCC/12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load binutils/2.39-GCCcore-12.2.0

# Activate the conda environment
source activate diffdsdfsim

# Export any other environment variables
export IGR_PATH=/home/grt22/diffsdfsim/IGR
export TORCH_CUDA_ARCH_LIST="7.5"
export PYTHONPATH=$PWD

