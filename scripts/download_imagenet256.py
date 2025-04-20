import os
import kagglehub

# TODO: Why is this saving in ~/.cache/kagglehub/datasets/dimensi0n/imagenet-256?

# Directory to store the dataset
data_dir = "data/imagenet256"

# Download the ImageNet 256x256 dataset ============================
def download_imagenet256(output_dir):
  os.makedirs(output_dir, exist_ok=True)
  print("Downloading ImageNet 256x256 dataset...")
  path = kagglehub.dataset_download("dimensi0n/imagenet-256")
  print("Path to dataset files:", path)

# Download and store the dataset ===============================================
if not os.path.exists(data_dir):
  download_imagenet256(data_dir)
else:
  print(f"ImageNet 256x256 dataset already exists in {data_dir}. Skipping download.")
