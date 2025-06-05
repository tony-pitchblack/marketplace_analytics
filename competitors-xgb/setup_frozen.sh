#!/usr/bin/env bash
# Verson for frozen dependencies 

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize micromamba for the current shell using the shell hook.
eval "$(micromamba shell hook -s bash)"

# Create env & install critical deps first
micromamba create -n competitors-xgb \
    python=3.12 \
    cython youtokentome -y # NOTE: always install first

# Install remaining deps
micromamba activate competitors-xgb 
pip install -r requirements_frozen.txt