#!/bin/bash

# Install rsync manually on instance first
# apt update && apt install -y rsync

set -e

echo "=== Step 1: Downloading and Installing Miniconda ==="
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh # gpu droplet
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh # macbook M1
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Initialize conda in bash (this adds necessary lines to ~/.bashrc)
~/miniconda3/bin/conda init bash
. ~/.bashrc

echo "Run 'source ~/.bashrc' manually to initialize the shell environment."
echo "After that, run 'bash tmp/setup2.sh' to continue the setup process."
