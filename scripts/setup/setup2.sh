# Load variables from .env file in the current directory
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found in the current directory."
    exit 1
fi

echo "=== Step 2: Creating the conda environment from environment.yaml ==="
cd $TARGET_VOLUME
conda env create --file=environment.yml

echo "=== Step 3: Aactivating the conda environment ==="
source ~/.bashrc
conda activate DiT-optimizations

echo "=== Step 4: Installing PyTorch ==="
pip3 install torch torchvision torchaudio
# GH200 lambda / cuda + ARM? (need to re-run separately)
# pip install torch==2.6 torchvision --index-url https://download.pytorch.org/whl/cu126

echo "Setup 5: Downloading data."
cd $TARGET_VOLUME && python scripts/download_cifar10.py

echo "Setup complete! Your environment is now ready."
