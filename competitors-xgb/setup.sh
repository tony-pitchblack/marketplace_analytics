#!/usr/bin/env bash

# micromamba create -n competitors-xgb \
#     python=3.12 cython youtokentome -y
# pip install -r requirements.txt

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize micromamba for the current shell using the shell hook.
eval "$(micromamba shell hook -s bash)"

# Activate your environment (replace "myenv" with your actual environment name)
micromamba activate competitors-xgb

# Install packages from requirements.txt using pip
pip install -r requirements.txt

echo "Setup complete!"
