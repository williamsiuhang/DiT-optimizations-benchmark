#!/bin/bash

set -e

# Install rsync manually on instance first
# apt update && apt install -y rsync
# Load variables from .env file in the current directory
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found in the current directory."
    exit 1
fi

echo "=== Step 1: Downloading and Installing Miniconda ==="
mkdir -p $TARGET_VOLUME/tmp/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $TARGET_VOLUME/tmp/miniconda3/miniconda.sh # gpu droplet
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O $TARGET_VOLUME/tmp/miniconda3/miniconda.sh # macbook M1
bash $TARGET_VOLUME/tmp/miniconda3/miniconda.sh -b -u -p $TARGET_VOLUME/tmp/miniconda3
rm $TARGET_VOLUME/tmp/miniconda3/miniconda.sh

# Initialize conda in bash (this adds necessary lines to ~/.bashrc)
$TARGET_VOLUME/tmp/miniconda3/bin/conda init bash
. ~/.bashrc

echo "Run 'source ~/.bashrc' manually to initialize the shell environment."
echo "After that, run 'bash scripts/setup/ubuntu/setup2.sh' to continue the setup process."
