#!/bin/bash

# Run text-to-image-to-3D pipeline in custom TRELLIS container

# Load necessary modules
module load gcc/11.2.0
module load cuda/12.2
module load python3
module load tacc-apptainer

# Ensure correct directory
cd $SCRATCH/x2sim

# Container path
export CONTAINER_PATH="$SCRATCH/x2sim/trellis.sif"

# Source environment variables
source credentials.sh

# Set your text prompt
export TEXT_PROMPT="A very simple bridge"

# Create cache directories in scratch space
mkdir -p $SCRATCH/cache/huggingface
mkdir -p $SCRATCH/cache/torch
mkdir -p $SCRATCH/cache/warp
mkdir -p $SCRATCH/cache/pip
mkdir -p $SCRATCH/cache/python

# Export cache environment variables
export HF_HOME=$SCRATCH/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export TRANSFORMERS_CACHE=$SCRATCH/cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub
export TORCH_HOME=$SCRATCH/cache/torch
export XDG_CACHE_HOME=$SCRATCH/cache
export PYTHONUSERBASE=$SCRATCH/python_env
export WARP_CACHE_PATH=$SCRATCH/cache/warp

# Run with environment variables and bind for caching and for running code example
apptainer exec --cleanenv --no-home --nv \
  --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" \
  --env "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE" \
  --env "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE" \
  --env "TORCH_HOME=$TORCH_HOME" \
  --env "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
  --env "PYTHONUSERBASE=$PYTHONUSERBASE" \
  --env "WARP_CACHE_PATH=$WARP_CACHE_PATH" \
  --env "SCRATCH=$SCRATCH" \
  --bind "$SCRATCH/cache:/tmp/cache" \
  --bind "$SCRATCH/cache/huggingface:/root/.cache/huggingface" \
  --bind "$SCRATCH/x2sim:/code" \
  --writable-tmpfs \
  $CONTAINER_PATH \
  /usr/bin/python3 /code/TRELLIS/run_x2sim.py "$TEXT_PROMPT"