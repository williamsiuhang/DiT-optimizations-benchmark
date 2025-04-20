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
echo "Do you want to download datasets? (1: CIFAR-10, 2: MSCOCO, 3: Skip)"
read -p "Enter your choice: " choice

case $choice in
    1)
        echo "Downloading CIFAR-10 dataset..."
        cd $TARGET_VOLUME && python scripts/download_cifar10.py
        ;;
    2)
        echo "Downloading MSCOCO dataset..."
        cd $TARGET_VOLUME && python scripts/download_mscoco.py
        ;;
    3)
        echo "Downloading ImageNet dataset..."
        cd $TARGET_VOLUME && python scripts/download_imagenet256.py
        ;;
    4)
        echo "Skipping dataset download."
        ;;
    *)
        echo "Invalid choice. Skipping dataset download."
        ;;
esac

echo "Setup complete! Your environment is now ready."
