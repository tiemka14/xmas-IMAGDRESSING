#!/usr/bin/env bash
set -euo pipefail

# Script: scripts/setup_conda_env.sh
# Purpose: Create conda env, install requirements, and optionally download model from HuggingFace

ENV_NAME=VITON_HD
PYTHON_VERSION=3.10.12
REQUIREMENTS_FILE=requirements.txt

echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "Activating conda environment: $ENV_NAME"
# Sourcing conda is necessary if not already initialized in the shell
if [ -f "$CONDA_EXE" ]; then
    # probably already configured in CI
    :
else
    # typical path for conda
    . "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi
conda activate "$ENV_NAME"

echo "Upgrading pip"
python -m pip install --upgrade pip

if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing python requirements from $REQUIREMENTS_FILE"
  pip install -r "$REQUIREMENTS_FILE"
else
  echo "Warning: $REQUIREMENTS_FILE not found"
fi

# Download models from Hugging Face
# The app/model.py loads the pretrained "IMAGDressing/IMAGDressing" model from Hugging Face.
# To manually download the model weights, you can:
# 1) Install huggingface-cli: pip install huggingface_hub
# 2) Log in: huggingface-cli login
# 3) Download or use from_pretrained in code: 
#    from diffusers import DiffusionPipeline
#    pipe = DiffusionPipeline.from_pretrained('IMAGDressing/IMAGDressing')

# Example: If you want to download model weights locally to /models/IMAGDressing:
# mkdir -p models/IMAGDressing
# huggingface-cli repo clone --depth=1 IMAGDressing/IMAGDressing models/IMAGDressing

# Note: you will need to accept model terms on Hugging Face if prompted or use a token.

echo "Conda env setup complete. Activate with: conda activate $ENV_NAME"