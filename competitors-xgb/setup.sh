#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize micromamba for the current shell using the shell hook.
eval "$(micromamba shell hook -s bash)"

# Create env & install critical deps first
micromamba create -n competitors-xgb \
    python=3.12 cython youtokentome -y

# Install remaining deps
micromamba activate competitors-xgb
pip install -r requirements.txt

echo "Setup complete!"
